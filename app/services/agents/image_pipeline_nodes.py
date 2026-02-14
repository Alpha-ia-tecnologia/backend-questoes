"""
Image Pipeline Nodes - Nós LangGraph para geração e validação de imagens.

Integra a geração de imagens diretamente no pipeline de questões,
com validação multimodal via Gemini Vision e retry automático.

⚡ Otimização: geração e validação em PARALELO via ThreadPoolExecutor.

Fluxo:
    [quality_gate] → image_router_decision → image_generator → image_validator → image_quality_router
                                                   ↑                                      ↓
                                                   └── retry (max 2) ←←←←←←←←←←←←←←←←←←←┘
"""

import logging
from typing import Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.services.agents.state import AgentState
from app.services.progress_manager import get_current_progress
from app.schemas.question_schema import QuestionSchema

logger = logging.getLogger(__name__)

MAX_IMAGE_RETRIES = 2
MAX_WORKERS = 4


def image_router_decision(state: AgentState) -> Literal["image_generator", "__end__"]:
    """
    Decide se deve gerar imagens ou finalizar.
    
    Rota para image_generator APENAS se image_dependency == "required".
    """
    query = state["query"]
    image_dep = getattr(query, "image_dependency", "none")
    
    if image_dep == "required":
        logger.info("🖼️ image_dependency=required → Roteando para geração de imagens")
        return "image_generator"
    else:
        logger.info(f"🖼️ image_dependency={image_dep} → Finalizando sem imagem")
        return "__end__"


def _generate_single_image(idx: int, q_data, corrections: str | None, retry_count: int) -> dict:
    """Gera imagem para uma única questão (thread-safe)."""
    from app.services.generate_image_agent_service import get_image_service
    
    try:
        if isinstance(q_data, dict):
            question = QuestionSchema(**q_data)
        else:
            question = q_data
        
        image_service = get_image_service()
        
        if corrections and retry_count > 0:
            logger.info(f"🔄 Regenerando imagem {idx} com correções: {corrections[:100]}...")
            result = image_service.generate_image_with_instructions(question, corrections)
        else:
            result = image_service.generate_image(question)
        
        logger.info(f"✅ Imagem gerada para questão {idx}")
        return {
            "question_index": idx,
            "image_base64": result.image_base64,
            "validation_status": "pending",
            "attempts": retry_count + 1,
            "corrections": None
        }
    except Exception as e:
        logger.error(f"❌ Erro ao gerar imagem para questão {idx}: {e}")
        return {
            "question_index": idx,
            "image_base64": None,
            "validation_status": "error",
            "attempts": retry_count + 1,
            "error": str(e),
            "corrections": None
        }


def image_generator_node(state: AgentState) -> dict:
    """
    Gera imagens para TODAS as questões do batch EM PARALELO.
    
    Usa ThreadPoolExecutor para disparar todas as gerações simultaneamente,
    reduzindo o tempo total de N*T para ~T (onde T é o tempo de 1 imagem).
    """
    progress = get_current_progress()
    questions = state.get("questions", [])
    existing_results = state.get("image_results") or []
    retry_count = state.get("image_retry_count", 0)
    
    if progress:
        progress.phase_start("image_generator", "Image Generation Agent", "🎨")
        progress.log("image_generator", f"Generating {len(questions)} images in parallel ⚡", "", "🖼️")
    
    # Separar questões que já têm imagem válida das que precisam gerar
    tasks_to_generate = []
    pre_validated = []
    
    for idx, q_data in enumerate(questions):
        existing = next(
            (r for r in existing_results if r.get("question_index") == idx and r.get("validation_status") == "valid"),
            None
        )
        if existing:
            logger.info(f"✅ Questão {idx}: Imagem já válida, pulando")
            pre_validated.append(existing)
            continue
        
        corrections = None
        failed_result = next(
            (r for r in existing_results if r.get("question_index") == idx and r.get("validation_status") == "invalid"),
            None
        )
        if failed_result:
            corrections = failed_result.get("corrections", "")
        
        tasks_to_generate.append((idx, q_data, corrections))
    
    # Gerar em paralelo
    image_results = list(pre_validated)
    
    if tasks_to_generate:
        workers = min(MAX_WORKERS, len(tasks_to_generate))
        logger.info(f"⚡ Disparando {len(tasks_to_generate)} gerações em paralelo ({workers} workers)")
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_generate_single_image, idx, q_data, corrections, retry_count): idx
                for idx, q_data, corrections in tasks_to_generate
            }
            
            for future in as_completed(futures):
                result = future.result()
                idx = result["question_index"]
                
                if progress:
                    status = "✅" if result.get("image_base64") else "❌"
                    title = ""
                    if idx < len(questions):
                        q = questions[idx]
                        title = (q.get("title", "") if isinstance(q, dict) else getattr(q, "title", ""))[:40]
                    progress.log("image_generator", f"{status} Image {idx + 1}: {title}...", "", "🎨")
                
                image_results.append(result)
    
    # Ordenar por índice para manter consistência
    image_results.sort(key=lambda r: r.get("question_index", 0))
    
    if progress:
        generated_count = sum(1 for r in image_results if r.get("image_base64"))
        progress.phase_end("image_generator", f"{generated_count}/{len(questions)} images generated ⚡")
    
    return {
        "image_results": image_results,
        "image_retry_count": retry_count
    }


def _validate_single_image(idx: int, q_data: dict, image_base64: str) -> dict:
    """Valida uma única imagem (thread-safe)."""
    from app.services.agents.image_validator_agent import get_image_validator_agent
    
    try:
        validator = get_image_validator_agent()
        validation = validator.validate(q_data, image_base64)
        
        is_valid = validation.get("valid", False)
        score = validation.get("score", 0)
        
        return {
            "question_index": idx,
            "validation_status": "valid" if is_valid else "invalid",
            "validation_score": score,
            "validation_issues": validation.get("issues", []),
            "corrections": validation.get("corrections", "") if not is_valid else None
        }
    except Exception as e:
        logger.error(f"❌ Erro ao validar imagem {idx}: {e}")
        return {
            "question_index": idx,
            "validation_status": "invalid",
            "validation_issues": [str(e)],
            "corrections": "Regenerar a imagem"
        }


def image_validator_node(state: AgentState) -> dict:
    """
    Valida TODAS as imagens geradas EM PARALELO usando Gemini Vision.
    """
    progress = get_current_progress()
    questions = state.get("questions", [])
    image_results = state.get("image_results", [])
    
    if progress:
        progress.phase_start("image_validator", "Image Validation Agent", "👁️")
    
    # Separar resultados que precisam validação dos que já estão prontos
    to_validate = []
    already_done = []
    
    for result in image_results:
        idx = result.get("question_index", 0)
        
        if result.get("validation_status") in ("valid", "error"):
            already_done.append(result)
            continue
        
        image_base64 = result.get("image_base64")
        if not image_base64 or idx >= len(questions):
            result["validation_status"] = "error"
            already_done.append(result)
            continue
        
        q_data = questions[idx]
        if not isinstance(q_data, dict):
            q_data = q_data.model_dump() if hasattr(q_data, 'model_dump') else q_data.__dict__
        
        to_validate.append((idx, q_data, image_base64, result))
    
    # Validar em paralelo
    validated_results = list(already_done)
    
    if to_validate:
        workers = min(MAX_WORKERS, len(to_validate))
        logger.info(f"⚡ Validando {len(to_validate)} imagens em paralelo ({workers} workers)")
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for idx, q_data, image_base64, original_result in to_validate:
                future = executor.submit(_validate_single_image, idx, q_data, image_base64)
                futures[future] = original_result
            
            for future in as_completed(futures):
                original_result = futures[future]
                validation = future.result()
                
                # Merge validation into original result (preserving image_base64)
                original_result.update(validation)
                
                if progress:
                    is_valid = validation.get("validation_status") == "valid"
                    score = validation.get("validation_score", 0)
                    icon = "✅" if is_valid else "❌"
                    idx = validation.get("question_index", 0)
                    progress.log(
                        "image_validator",
                        f"{icon} Image {idx + 1}: {'Approved' if is_valid else 'Rejected'} (score: {score})",
                        ", ".join(validation.get("validation_issues", [])) if not is_valid else "",
                        icon
                    )
                
                validated_results.append(original_result)
    
    # Ordenar por índice
    validated_results.sort(key=lambda r: r.get("question_index", 0))
    
    valid_count = sum(1 for r in validated_results if r.get("validation_status") == "valid")
    invalid_count = sum(1 for r in validated_results if r.get("validation_status") == "invalid")
    
    if progress:
        progress.phase_end(
            "image_validator",
            f"✅ {valid_count} approved, ❌ {invalid_count} rejected"
        )
    
    return {"image_results": validated_results}


def image_quality_router(state: AgentState) -> Literal["image_generator", "__end__"]:
    """
    Decide se precisa regenerar imagens ou finalizar.
    
    - Se há imagens inválidas E attempts < MAX_IMAGE_RETRIES → retry
    - Caso contrário → finalizar (imagens inválidas ficam com needs_manual_image=true)
    """
    image_results = state.get("image_results", [])
    retry_count = state.get("image_retry_count", 0)
    
    invalid_images = [r for r in image_results if r.get("validation_status") == "invalid"]
    
    if invalid_images and retry_count < MAX_IMAGE_RETRIES:
        logger.info(
            f"🔄 Retry de imagem #{retry_count + 1}/{MAX_IMAGE_RETRIES}: "
            f"{len(invalid_images)} imagens inválidas"
        )
        
        progress = get_current_progress()
        if progress:
            progress.retry(
                retry_count + 1,
                f"{len(invalid_images)} images rejected, retrying..."
            )
        
        return "image_generator"
    
    if invalid_images:
        logger.warning(
            f"⚠️ {len(invalid_images)} imagens ainda inválidas após {retry_count} tentativas. "
            f"Habilitando geração manual."
        )
    else:
        logger.info("✅ Todas as imagens validadas com sucesso!")
    
    return "__end__"


def increment_image_retry(state: AgentState) -> dict:
    """Incrementa o contador de retry de imagem antes de regenerar."""
    return {"image_retry_count": state.get("image_retry_count", 0) + 1}
