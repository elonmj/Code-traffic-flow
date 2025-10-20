"""
StructuredLogger - Logging structuré avec structlog

Implémente Innovation 7: Dual Logging (fichier + console)
- Fichier: JSON structuré pour analyse automatisée
- Console: Formaté lisible pour développeurs

Logging structuré permet:
- Recherche événements spécifiques (cache_hit, checkpoint_saved)
- Agrégation métriques (temps entraînement, GPU utilisé)
- Debugging précis avec contexte complet
"""

import logging
import sys
from pathlib import Path
from typing import Any, Optional
import structlog

from domain.interfaces import Logger


class StructuredLogger(Logger):
    """Logger structuré avec dual output (Innovation 7)."""
    
    def __init__(
        self,
        log_file: Optional[Path] = None,
        log_level: str = "INFO"
    ):
        """Initialise logger structuré.
        
        Args:
            log_file: Chemin fichier log (None = console uniquement)
            log_level: Niveau logging ("DEBUG", "INFO", "WARNING", "ERROR")
        """
        self.log_file = Path(log_file) if log_file else None
        self.log_level = getattr(logging, log_level.upper())
        
        # Configuration structlog
        self._configure_structlog()
        
        # Logger structuré
        self.logger = structlog.get_logger()
    
    def _configure_structlog(self) -> None:
        """Configure structlog avec processors et outputs."""
        
        # Processors communs
        shared_processors = [
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]
        
        # Configuration logging standard Python
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=self.log_level,
        )
        
        # Handlers
        handlers = []
        
        # Handler console: formaté lisible
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        handlers.append(console_handler)
        
        # Handler fichier: JSON structuré
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(self.log_level)
            handlers.append(file_handler)
        
        # Configuration structlog
        structlog.configure(
            processors=shared_processors + [
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Configuration formatters
        console_formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=True),
            foreign_pre_chain=shared_processors,
        )
        
        json_formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
            foreign_pre_chain=shared_processors,
        )
        
        # Application formatters aux handlers
        console_handler.setFormatter(console_formatter)
        if self.log_file:
            file_handler.setFormatter(json_formatter)
        
        # Configuration logger racine
        root_logger = logging.getLogger()
        root_logger.handlers = handlers
        root_logger.setLevel(self.log_level)
    
    def info(self, event: str, **kwargs: Any) -> None:
        """Log événement niveau INFO avec contexte structuré.
        
        Args:
            event: Nom événement (ex: "cache_baseline_hit")
            **kwargs: Contexte additionnel (clés-valeurs)
        """
        self.logger.info(event, **kwargs)
    
    def warning(self, event: str, **kwargs: Any) -> None:
        """Log événement niveau WARNING avec contexte structuré.
        
        Args:
            event: Nom événement
            **kwargs: Contexte additionnel
        """
        self.logger.warning(event, **kwargs)
    
    def error(self, event: str, **kwargs: Any) -> None:
        """Log événement niveau ERROR avec contexte structuré.
        
        Args:
            event: Nom événement
            **kwargs: Contexte additionnel
        """
        self.logger.error(event, **kwargs)
    
    def exception(self, event: str, **kwargs: Any) -> None:
        """Log exception avec traceback complet.
        
        Args:
            event: Nom événement
            **kwargs: Contexte additionnel
        """
        self.logger.exception(event, **kwargs)
    
    def debug(self, event: str, **kwargs: Any) -> None:
        """Log événement niveau DEBUG avec contexte structuré.
        
        Args:
            event: Nom événement
            **kwargs: Contexte additionnel
        """
        self.logger.debug(event, **kwargs)


# Événements structurés prédéfinis (documentation)
# Cette liste documente les événements clés pour recherche/analyse

STANDARD_EVENTS = {
    # Cache events
    "cache_baseline_hit": "Cache baseline trouvé",
    "cache_baseline_miss": "Cache baseline non trouvé",
    "cache_baseline_saved": "Cache baseline sauvegardé",
    "cache_rl_hit": "Cache RL trouvé",
    "cache_rl_miss": "Cache RL non trouvé",
    "cache_corrupted": "Cache corrompu détecté",
    
    # Checkpoint events
    "checkpoint_saved": "Checkpoint sauvegardé",
    "checkpoint_loaded": "Checkpoint chargé",
    "checkpoint_deleted_rotation": "Checkpoint supprimé (rotation)",
    "checkpoint_rotation_complete": "Rotation checkpoints terminée",
    "no_compatible_checkpoint_found": "Aucun checkpoint compatible",
    
    # Training events
    "rl_training_starting": "Entraînement RL démarré",
    "rl_training_complete": "Entraînement RL terminé",
    "rl_evaluation_complete": "Évaluation RL terminée",
    "baseline_simulation_starting": "Simulation baseline démarrée",
    "baseline_simulation_complete": "Simulation baseline terminée",
    
    # Scenario events
    "scenario_execution_starting": "Scénario démarré",
    "scenario_execution_complete": "Scénario terminé",
    "multiple_scenarios_execution_starting": "Exécution multi-scénarios démarrée",
    "multiple_scenarios_execution_complete": "Exécution multi-scénarios terminée",
    
    # Comparison events
    "comparison_computed": "Comparaison baseline vs RL calculée",
    
    # Error events
    "checkpoint_save_failed": "Échec sauvegarde checkpoint",
    "checkpoint_load_failed": "Échec chargement checkpoint",
}
