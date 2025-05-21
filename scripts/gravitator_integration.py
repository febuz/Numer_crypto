#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integratie van de Data Gravitator met de bestaande Numerai pijplijn.

Dit script voegt de Data Gravitator toe als een stap in de Numerai pijplijn
en stelt commando's in om de gravitator functies aan te roepen.
"""

import os
import sys
import argparse
from pathlib import Path
import logging
import subprocess
from datetime import datetime
import json
from typing import Dict, List, Optional, Union, Any

# Voeg root directory toe aan Python path
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
root_dir = script_dir.parent
sys.path.append(str(root_dir))

# Interne imports
from utils.pipeline.gravitator import DataGravitator
from utils.log_utils import setup_logging


def log_parameters(
    logger: logging.Logger,
    params: Dict[str, Any],
    section: str = "Parameters"
) -> None:
    """
    Log parameters in een gestructureerd format.
    
    Args:
        logger: Logger om mee te loggen
        params: Dictionary met parameters
        section: Naam van de sectie voor de log
    """
    logger.info(f"--- {section} ---")
    for key, value in sorted(params.items()):
        logger.info(f"  {key}: {value}")
    logger.info("-" * (len(section) + 8))


def create_status_file(
    status_dir: Union[str, Path],
    process_name: str,
    status: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Maak een statusbestand voor monitoring.
    
    Args:
        status_dir: Directory voor statusbestanden
        process_name: Naam van het proces
        status: Statustekst (bijv. 'started', 'completed', 'failed')
        metadata: Extra metadata om op te slaan
        
    Returns:
        Pad naar het gemaakte statusbestand
    """
    if isinstance(status_dir, str):
        status_dir = Path(status_dir)
    
    # Zorg dat de directory bestaat
    status_dir.mkdir(exist_ok=True, parents=True)
    
    # Maak status data
    status_data = {
        'process': process_name,
        'status': status,
        'timestamp': datetime.now().isoformat(),
        'hostname': os.uname().nodename
    }
    
    # Voeg metadata toe indien opgegeven
    if metadata is not None:
        status_data['metadata'] = metadata
    
    # Maak bestandsnaam
    status_file = status_dir / f"{process_name}_{status}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Schrijf naar bestand
    with open(status_file, 'w') as f:
        json.dump(status_data, f, indent=2)
    
    return str(status_file)


def add_gravitator_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Voeg Data Gravitator argumenten toe aan een bestaande argument parser.
    
    Args:
        parser: Bestaande ArgumentParser
        
    Returns:
        De aangepaste ArgumentParser
    """
    # Data Gravitator groep
    gravitator_group = parser.add_argument_group('Data Gravitator opties')
    
    gravitator_group.add_argument(
        '--use-gravitator',
        action='store_true',
        help='Gebruik de Data Gravitator voor signaalverwerking'
    )
    
    gravitator_group.add_argument(
        '--gravitator-models-dir',
        type=str,
        default=None,
        help='Directory met modeloutputs voor de gravitator (standaard: {base_dir}/prediction)'
    )
    
    gravitator_group.add_argument(
        '--gravitator-output-dir',
        type=str,
        default=None,
        help='Map voor gravitator output bestanden (standaard: {base_dir}/gravitator)'
    )
    
    gravitator_group.add_argument(
        '--gravitator-target-col',
        type=str,
        default='target',
        help='Kolomnaam van het doelkenmerk'
    )
    
    gravitator_group.add_argument(
        '--gravitator-ensemble-method',
        type=str,
        default='mean_rank',
        choices=['mean', 'median', 'mean_rank', 'mean_weighted'],
        help='Methode voor ensemble creatie'
    )
    
    gravitator_group.add_argument(
        '--gravitator-selection-method',
        type=str,
        default='combined_rank',
        choices=['combined_rank', 'threshold', 'pareto'],
        help='Methode voor signaalselectie'
    )
    
    gravitator_group.add_argument(
        '--gravitator-top-n',
        type=int,
        default=50,
        help='Aantal signalen om te selecteren'
    )
    
    gravitator_group.add_argument(
        '--gravitator-min-ic',
        type=float,
        default=0.01,
        help='Minimum IC voor opname'
    )
    
    gravitator_group.add_argument(
        '--gravitator-min-sharpe',
        type=float,
        default=0.5,
        help='Minimum Sharpe ratio voor opname'
    )
    
    gravitator_group.add_argument(
        '--gravitator-no-neutralize',
        action='store_true',
        help='Schakel signaal neutralisatie uit'
    )
    
    gravitator_group.add_argument(
        '--gravitator-submit',
        action='store_true',
        help='Automatisch indienen bij Numerai na creatie'
    )
    
    gravitator_group.add_argument(
        '--today-only',
        action='store_true',
        default=True,
        help='Alleen voorspellingen van vandaag laden (standaard: True)'
    )
    
    gravitator_group.add_argument(
        '--all-predictions',
        action='store_true',
        help='Gebruik alle voorspellingsbestanden (niet alleen van vandaag)'
    )
    
    gravitator_group.add_argument(
        '--ensure-all-universe-symbols',
        action='store_true',
        default=True,
        help='Zorg dat alle universe symbolen worden opgenomen in de submission (standaard aan)'
    )
    
    gravitator_group.add_argument(
        '--symbol-alignment',
        type=str,
        default=None,
        help='Pad naar symbol alignment JSON bestand'
    )
    
    return parser


def run_gravitator(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Voer de Data Gravitator pipeline uit op basis van commandoregel argumenten.
    
    Args:
        args: CommandArgs namespace met gravitator instellingen
        
    Returns:
        Dictionary met resultaten
    """
    # Bepaal basismap
    base_dir = Path(args.base_dir if hasattr(args, 'base_dir') else '/media/knight2/EDB/numer_crypto_temp')
    
    # Bepaal tournooitype
    tournament = args.tournament if hasattr(args, 'tournament') else 'crypto'
    
    # Always use real targets - no synthetic target generation allowed
    
    # Setup logging
    log_dir = base_dir / 'log' / 'gravitator'
    log_dir.mkdir(exist_ok=True, parents=True)
    
    logger = setup_logging(
        name="gravitator_integration",
        level=logging.INFO,
        create_file=True,
        log_to_markdown=False
    )
    
    logger.info("Starten Data Gravitator integratie...")
    
    # Log parameters
    params = {
        'base_dir': str(base_dir),
        'tournament': tournament,
        'models_dir': args.gravitator_models_dir if hasattr(args, 'gravitator_models_dir') and args.gravitator_models_dir else str(base_dir / 'prediction'),
        'output_dir': args.gravitator_output_dir if hasattr(args, 'gravitator_output_dir') and args.gravitator_output_dir else str(base_dir / 'gravitator'),
        'target_col': args.gravitator_target_col if hasattr(args, 'gravitator_target_col') else 'target',
        'ensemble_method': args.gravitator_ensemble_method if hasattr(args, 'gravitator_ensemble_method') else 'mean_rank',
        'selection_method': args.gravitator_selection_method if hasattr(args, 'gravitator_selection_method') else 'combined_rank',
        'top_n': args.gravitator_top_n if hasattr(args, 'gravitator_top_n') else 50,
        'min_ic': args.gravitator_min_ic if hasattr(args, 'gravitator_min_ic') else 0.01,
        'min_sharpe': args.gravitator_min_sharpe if hasattr(args, 'gravitator_min_sharpe') else 0.5,
        'neutralize': not (hasattr(args, 'gravitator_no_neutralize') and args.gravitator_no_neutralize),
        'submit': hasattr(args, 'gravitator_submit') and args.gravitator_submit,
        'today_only': not (hasattr(args, 'all_predictions') and args.all_predictions),
        'include_live_universe': not hasattr(args, 'ensure_all_universe_symbols') or args.ensure_all_universe_symbols
    }
    
    # Check if symbol alignment file is provided
    symbol_alignment_file = None
    if hasattr(args, 'symbol_alignment') and args.symbol_alignment:
        symbol_alignment_file = args.symbol_alignment
        logger.info(f"Symbol alignment bestand opgegeven: {symbol_alignment_file}")
        params['symbol_alignment_file'] = symbol_alignment_file
    
    log_parameters(logger, params, "Data Gravitator Parameters")
    
    # Maak status bestand aan
    status_dir = base_dir / 'status'
    status_dir.mkdir(exist_ok=True, parents=True)
    
    start_status = create_status_file(
        status_dir=status_dir,
        process_name='data_gravitator',
        status='started',
        metadata=params
    )
    
    logger.info(f"Status bestand aangemaakt: {start_status}")
    
    try:
        # Load symbol alignment if provided
        symbol_mapping = None
        if symbol_alignment_file and os.path.exists(symbol_alignment_file):
            try:
                logger.info(f"Laden van symbol alignment data uit {symbol_alignment_file}")
                with open(symbol_alignment_file, 'r') as f:
                    symbol_mapping = json.load(f)
                logger.info(f"Symbol mapping geladen met {len(symbol_mapping)} symbolen")
            except Exception as e:
                logger.warning(f"Fout bij laden van symbol alignment: {str(e)}")
        
        # Stel de Data Gravitator in
        gravitator = DataGravitator(
            base_dir=params['base_dir'],
            output_dir=params['output_dir'],
            log_level=logging.INFO,
            min_ic_threshold=params['min_ic'],
            min_sharpe_threshold=params['min_sharpe'],
            neutralize=params['neutralize'],
            tournament=params['tournament']
        )
        
        # Always using real targets - synthetic target generation is disabled
        
        # Voer de volledige pijplijn uit
        submission_path = gravitator.run_full_pipeline(
            models_dir=params['models_dir'],
            target_col=params['target_col'],
            ensemble_method=params['ensemble_method'],
            selection_method=params['selection_method'],
            top_n=params['top_n'],
            auto_submit=params.get('submit', False),
            today_only=params.get('today_only', True),
            include_live_universe=params.get('include_live_universe', True)  # Include all live universe symbols by default
        )
        
        # Bewaar een metriekenrapport
        metrics_path = gravitator.save_metrics_report()
        
        # Maak een voltooiingsstatus bestand aan
        results = {
            'submission_path': submission_path,
            'metrics_path': metrics_path,
            'selected_signals': gravitator.selected_signals,
            'processed_models': gravitator.processed_models
        }
        
        # Add symbol mapping info to results if available
        if symbol_mapping:
            results['symbol_mapping_file'] = symbol_alignment_file
            results['symbol_mapping_count'] = len(symbol_mapping)
        
        completion_status = create_status_file(
            status_dir=status_dir,
            process_name='data_gravitator',
            status='completed',
            metadata={
                **params,
                **results
            }
        )
        
        logger.info(f"Data Gravitator succesvol voltooid")
        logger.info(f"Submission pad: {submission_path}")
        logger.info(f"Metrics rapport: {metrics_path}")
        logger.info(f"Status bestand: {completion_status}")
        
        return {
            'success': True,
            **results
        }
        
    except Exception as e:
        # Log fout
        logger.error(f"Fout bij uitvoeren van Data Gravitator: {str(e)}", exc_info=True)
        
        # Maak fout status bestand aan
        error_status = create_status_file(
            status_dir=status_dir,
            process_name='data_gravitator',
            status='failed',
            metadata={
                **params,
                'error': str(e)
            }
        )
        
        logger.info(f"Fout status bestand: {error_status}")
        
        return {
            'success': False,
            'error': str(e),
            'error_status': error_status
        }


def integrate_with_pipeline(pipeline_script: str) -> None:
    """
    Integreer de Data Gravitator met de bestaande pijplijn.
    
    Args:
        pipeline_script: Pad naar het pijplijn script (go_pipeline.sh)
    """
    # Controleer of het script bestaat
    if not os.path.exists(pipeline_script):
        print(f"Fout: Pijplijn script niet gevonden: {pipeline_script}")
        sys.exit(1)
    
    print(f"Integreren van Data Gravitator met pijplijn: {pipeline_script}")
    
    # Lees het script
    with open(pipeline_script, 'r') as f:
        script_content = f.read()
    
    # Controleer of de gravitator al is geïntegreerd
    if "run_gravitator()" in script_content:
        print("Data Gravitator is al geïntegreerd in de pijplijn")
        return
    
    # Functie definitie om toe te voegen
    gravitator_function = """
# Data Gravitator functie
run_gravitator() {
    log_info "Uitvoeren van Data Gravitator voor signaalverwerking"
    
    if [ "$USE_GRAVITATOR" != "true" ]; then
        log_info "Data Gravitator overgeslagen (--use-gravitator niet opgegeven)"
        return 0
    fi
    
    # Check for symbol alignment file
    SYMBOL_ALIGNMENT_ARGS=""
    if [ -f "${BASE_DIR}/data/features/symbol_alignment.json" ]; then
        log_info "Using symbol alignment information from feature generation"
        SYMBOL_ALIGNMENT_ARGS="--symbol-alignment ${BASE_DIR}/data/features/symbol_alignment.json"
    fi
    
    # Voer Data Gravitator integratie script uit
    python3 "$SCRIPT_DIR/scripts/gravitator_integration.py" \\
        --base-dir "$BASE_DIR" \\
        --tournament "$TOURNAMENT" \\
        --gravitator-models-dir "${GRAVITATOR_MODELS_DIR:-$PREDICTION_DIR}" \\
        --gravitator-output-dir "${GRAVITATOR_OUTPUT_DIR:-$BASE_DIR/gravitator}" \\
        --gravitator-target-col "${GRAVITATOR_TARGET_COL:-target}" \\
        --gravitator-ensemble-method "${GRAVITATOR_ENSEMBLE_METHOD:-mean_rank}" \\
        --gravitator-selection-method "${GRAVITATOR_SELECTION_METHOD:-combined_rank}" \\
        --gravitator-top-n "${GRAVITATOR_TOP_N:-50}" \\
        --gravitator-min-ic "${GRAVITATOR_MIN_IC:-0.01}" \\
        --gravitator-min-sharpe "${GRAVITATOR_MIN_SHARPE:-0.5}" \\
        ${GRAVITATOR_NO_NEUTRALIZE:+--gravitator-no-neutralize} \\
        $SYMBOL_ALIGNMENT_ARGS
    
    if [ $? -ne 0 ]; then
        log_error "Data Gravitator uitvoering mislukt"
        return 1
    fi
    
    log_info "Data Gravitator succesvol uitgevoerd"
    return 0
}
"""
    
    # Variabelen om toe te voegen aan de parse_args functie
    gravitator_vars = """
    # Data Gravitator variabelen
    USE_GRAVITATOR=false
    GRAVITATOR_MODELS_DIR=""
    GRAVITATOR_OUTPUT_DIR=""
    GRAVITATOR_TARGET_COL="target"
    GRAVITATOR_ENSEMBLE_METHOD="mean_rank"
    GRAVITATOR_SELECTION_METHOD="combined_rank"
    GRAVITATOR_TOP_N=50
    GRAVITATOR_MIN_IC=0.01
    GRAVITATOR_MIN_SHARPE=0.5
    GRAVITATOR_NO_NEUTRALIZE=false
"""
    
    # Argumenten om toe te voegen aan de parse_args functie
    gravitator_args = """
        --use-gravitator)
            USE_GRAVITATOR=true
            shift
            ;;
        --gravitator-models-dir)
            GRAVITATOR_MODELS_DIR="$2"
            shift 2
            ;;
        --gravitator-output-dir)
            GRAVITATOR_OUTPUT_DIR="$2"
            shift 2
            ;;
        --gravitator-target-col)
            GRAVITATOR_TARGET_COL="$2"
            shift 2
            ;;
        --gravitator-ensemble-method)
            GRAVITATOR_ENSEMBLE_METHOD="$2"
            shift 2
            ;;
        --gravitator-selection-method)
            GRAVITATOR_SELECTION_METHOD="$2"
            shift 2
            ;;
        --gravitator-top-n)
            GRAVITATOR_TOP_N="$2"
            shift 2
            ;;
        --gravitator-min-ic)
            GRAVITATOR_MIN_IC="$2"
            shift 2
            ;;
        --gravitator-min-sharpe)
            GRAVITATOR_MIN_SHARPE="$2"
            shift 2
            ;;
        --gravitator-no-neutralize)
            GRAVITATOR_NO_NEUTRALIZE=true
            shift
            ;;
"""
    
    # Help tekst om toe te voegen
    gravitator_help = """
  Data Gravitator opties:
    --use-gravitator                 Gebruik de Data Gravitator voor signaalverwerking
    --gravitator-models-dir DIR      Directory met modeloutputs voor de gravitator
    --gravitator-output-dir DIR      Map voor gravitator output bestanden
    --gravitator-target-col COL      Kolomnaam van het doelkenmerk
    --gravitator-ensemble-method M   Methode voor ensemble creatie
                                     (mean, median, mean_rank, mean_weighted)
    --gravitator-selection-method M  Methode voor signaalselectie
                                     (combined_rank, threshold, pareto)
    --gravitator-top-n N             Aantal signalen om te selecteren
    --gravitator-min-ic VALUE        Minimum IC voor opname
    --gravitator-min-sharpe VALUE    Minimum Sharpe ratio voor opname
    --gravitator-no-neutralize       Schakel signaal neutralisatie uit
"""
    
    # Voeg functie toe na de laatste functie definitie
    last_function_index = script_content.rfind("}")
    if last_function_index != -1:
        updated_content = (
            script_content[:last_function_index+1] + 
            "\n" + gravitator_function + 
            script_content[last_function_index+1:]
        )
    else:
        # Fallback: voeg toe aan einde van bestand
        updated_content = script_content + "\n" + gravitator_function
    
    # Voeg variabelen toe aan parse_args functie
    parse_args_index = updated_content.find("parse_args()")
    if parse_args_index != -1:
        # Zoek variabele sectie
        vars_end_index = updated_content.find("while", parse_args_index)
        if vars_end_index != -1:
            updated_content = (
                updated_content[:vars_end_index] + 
                gravitator_vars + 
                updated_content[vars_end_index:]
            )
    
    # Voeg argumenten toe aan de parse_args functie
    case_index = updated_content.find("case \"$1\" in", parse_args_index)
    if case_index != -1:
        # Zoek einde van case statement maar voor esac
        esac_index = updated_content.find("esac", case_index)
        if esac_index != -1:
            # Voeg argumenten toe voor esac
            updated_content = (
                updated_content[:esac_index] + 
                gravitator_args + 
                updated_content[esac_index:]
            )
    
    # Voeg help tekst toe
    usage_index = updated_content.find("show_usage()")
    if usage_index != -1:
        # Zoek einde van functie
        usage_end = updated_content.find("}", usage_index)
        if usage_end != -1:
            # Voeg help tekst toe voor einde van functie
            updated_content = (
                updated_content[:usage_end] + 
                gravitator_help + 
                updated_content[usage_end:]
            )
    
    # Voeg gravitator aanroep toe aan de run_pipeline functie
    run_pipeline_index = updated_content.find("run_pipeline()")
    if run_pipeline_index != -1:
        # Zoek einde van run_pipeline functie
        pipeline_end = updated_content.find("}", run_pipeline_index)
        
        if pipeline_end != -1:
            # Bereken positie voor invoeging (voor de laatste return)
            return_index = updated_content.rfind("return", run_pipeline_index, pipeline_end)
            
            if return_index != -1:
                pipeline_call = """
    # Voer Data Gravitator uit als laatste stap voor submissie
    run_gravitator || return 1
"""
                updated_content = (
                    updated_content[:return_index] + 
                    pipeline_call + 
                    updated_content[return_index:]
                )
    
    # Schrijf bijgewerkt script
    with open(pipeline_script, 'w') as f:
        f.write(updated_content)
    
    print(f"Data Gravitator succesvol geïntegreerd in pijplijn: {pipeline_script}")


def main():
    """
    Hoofdfunctie voor gravitator_integration.py
    """
    parser = argparse.ArgumentParser(description="Data Gravitator Integratie")
    
    # Algemene argumenten
    parser.add_argument('--base-dir', type=str, default='/media/knight2/EDB/numer_crypto_temp',
                        help='Basismap voor alle data')
    parser.add_argument('--tournament', type=str, default='crypto', choices=['crypto', 'signals'],
                        help='Welke Numerai competitie (crypto of signals)')
    
    # Integratie argumenten
    parser.add_argument('--integrate', action='store_true',
                        help='Integreer met de bestaande pijplijn')
    parser.add_argument('--pipeline-script', type=str, default=None,
                        help='Pad naar het pijplijn script (go_pipeline.sh)')
    
    # Voeg gravitator argumenten toe
    parser = add_gravitator_args(parser)
    
    args = parser.parse_args()
    
    # Integreren met bestaande pijplijn indien gevraagd
    if args.integrate:
        # Bepaal het pad naar het pijplijn script
        pipeline_script = args.pipeline_script
        if pipeline_script is None:
            # Default pad
            pipeline_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'go_pipeline.sh')
        
        # Voer integratie uit
        integrate_with_pipeline(pipeline_script)
        return
    
    # Anders voer gravitator uit
    run_gravitator(args)


if __name__ == "__main__":
    main()