#!/usr/bin/env python3
"""
Interface en ligne de commande pour le pipeline FACS
Usage: python facs_cli.py [commande] [options]
"""

import argparse
import sys
from pathlib import Path
from facs_autogating import FCSGatingPipeline, example_standard_workflow
from facs_workflows_advanced import BatchFCSAnalysis
from facs_utilities import FCSValidator, ChannelDetector, auto_suggest_workflow


def cmd_analyze(args):
    """Analyse un fichier FCS"""
    print(f"\n{'='*60}")
    print(f"ANALYSE FACS: {Path(args.input).name}")
    print(f"{'='*60}\n")
    
    try:
        # Workflow standard
        pipeline, stats = example_standard_workflow(
            args.input,
            args.output_dir
        )
        
        print("\n📊 Résultats:")
        print(stats[['Population', 'Count', 'Percentage_of_total']].to_string(index=False))
        
        print(f"\n✅ Analyse terminée avec succès!")
        print(f"📁 Résultats dans: {args.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        sys.exit(1)


def cmd_batch(args):
    """Analyse par lot"""
    print(f"\n{'='*60}")
    print(f"ANALYSE PAR LOT")
    print(f"{'='*60}\n")
    
    # Lire la liste de fichiers
    with open(args.file_list, 'r') as f:
        fcs_files = [line.strip() for line in f if line.strip()]
    
    print(f"📁 {len(fcs_files)} fichiers à traiter")
    
    # Noms d'échantillons
    sample_names = [Path(f).stem for f in fcs_files]
    
    try:
        # Analyse par lot
        batch = BatchFCSAnalysis(fcs_files, sample_names)
        pipelines = batch.run_standard_pipeline(
            compensate=not args.no_compensation,
            transform=args.transform,
            gate_strategy=args.strategy
        )
        
        # Comparaison
        comparison = batch.compare_populations()
        
        # Export
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        batch.export_comparative_excel(str(output_dir / 'batch_analysis.xlsx'))
        batch.plot_comparative_barplot(save_path=str(output_dir / 'comparison.png'))
        
        print(f"\n✅ Analyse par lot terminée!")
        print(f"📁 Résultats dans: {args.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        sys.exit(1)


def cmd_validate(args):
    """Valide des fichiers FCS"""
    print(f"\n{'='*60}")
    print(f"VALIDATION DE FICHIERS FCS")
    print(f"{'='*60}\n")
    
    # Lire la liste
    if args.file_list:
        with open(args.file_list, 'r') as f:
            fcs_files = [line.strip() for line in f if line.strip()]
    else:
        fcs_files = [args.input]
    
    # Validation
    results = FCSValidator.batch_validate_fcs(fcs_files)
    
    # Affichage
    print(results.to_string(index=False))
    
    # Statistiques
    n_valid = results['Valid'].sum()
    n_total = len(results)
    
    print(f"\n📊 Résumé:")
    print(f"   • Fichiers valides: {n_valid}/{n_total}")
    print(f"   • Fichiers invalides: {n_total - n_valid}/{n_total}")
    
    if args.output:
        results.to_excel(args.output, index=False)
        print(f"\n💾 Rapport sauvegardé: {args.output}")


def cmd_suggest(args):
    """Suggère un workflow pour un fichier FCS"""
    print(f"\n{'='*60}")
    print(f"SUGGESTION DE WORKFLOW")
    print(f"{'='*60}\n")
    
    try:
        # Analyser et suggérer
        code = auto_suggest_workflow(args.input)
        
        print(code)
        
        # Sauvegarder si demandé
        if args.output:
            with open(args.output, 'w') as f:
                f.write(code)
            print(f"\n💾 Workflow sauvegardé: {args.output}")
            
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        sys.exit(1)


def cmd_channels(args):
    """Liste les canaux d'un fichier FCS"""
    print(f"\n{'='*60}")
    print(f"ANALYSE DES CANAUX")
    print(f"{'='*60}\n")
    
    try:
        from facs_autogating import FCSGatingPipeline
        
        pipeline = FCSGatingPipeline(args.input, compensate=False, transform=None)
        
        print(f"📁 Fichier: {Path(args.input).name}")
        print(f"📊 {len(pipeline.data):,} événements")
        print(f"📋 {len(pipeline.channels)} canaux\n")
        
        # Détection automatique
        detected = ChannelDetector.detect_channels(pipeline)
        
        if detected:
            print("🔍 Marqueurs détectés:\n")
            for category, channels in detected.items():
                print(f"  {category}:")
                for ch in channels:
                    print(f"    • {ch}")
                print()
        
        # Suggestions
        suggestions = ChannelDetector.suggest_gating_strategy(detected)
        
        if suggestions:
            print("💡 Stratégie de gating suggérée:\n")
            for suggestion in suggestions:
                print(f"  {suggestion}")
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        sys.exit(1)


def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(
        description='Pipeline d\'automatisation du gating FACS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:

  # Analyser un fichier
  python facs_cli.py analyze sample.fcs -o ./results

  # Analyse par lot
  python facs_cli.py batch -l file_list.txt -o ./results

  # Valider des fichiers
  python facs_cli.py validate sample.fcs
  python facs_cli.py validate -l file_list.txt -o validation_report.xlsx

  # Suggérer un workflow
  python facs_cli.py suggest sample.fcs -o suggested_workflow.py

  # Lister les canaux
  python facs_cli.py channels sample.fcs
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commandes disponibles')
    
    # Commande: analyze
    parser_analyze = subparsers.add_parser('analyze', help='Analyser un fichier FCS')
    parser_analyze.add_argument('input', help='Fichier FCS d\'entrée')
    parser_analyze.add_argument('-o', '--output-dir', default='./results',
                               help='Répertoire de sortie (défaut: ./results)')
    parser_analyze.set_defaults(func=cmd_analyze)
    
    # Commande: batch
    parser_batch = subparsers.add_parser('batch', help='Analyse par lot')
    parser_batch.add_argument('-l', '--file-list', required=True,
                             help='Fichier texte contenant la liste des FCS (un par ligne)')
    parser_batch.add_argument('-o', '--output-dir', default='./results',
                             help='Répertoire de sortie')
    parser_batch.add_argument('--no-compensation', action='store_true',
                             help='Désactiver la compensation spectrale')
    parser_batch.add_argument('--transform', default='logicle',
                             choices=['logicle', 'asinh', 'hyperlog', 'none'],
                             help='Type de transformation')
    parser_batch.add_argument('--strategy', default='standard',
                             choices=['standard', 'lymphocytes'],
                             help='Stratégie de gating')
    parser_batch.set_defaults(func=cmd_batch)
    
    # Commande: validate
    parser_validate = subparsers.add_parser('validate', help='Valider des fichiers FCS')
    group = parser_validate.add_mutually_exclusive_group(required=True)
    group.add_argument('input', nargs='?', help='Fichier FCS à valider')
    group.add_argument('-l', '--file-list',
                      help='Fichier texte avec liste de FCS')
    parser_validate.add_argument('-o', '--output',
                                help='Sauvegarder le rapport (Excel)')
    parser_validate.set_defaults(func=cmd_validate)
    
    # Commande: suggest
    parser_suggest = subparsers.add_parser('suggest',
                                          help='Suggérer un workflow adapté')
    parser_suggest.add_argument('input', help='Fichier FCS à analyser')
    parser_suggest.add_argument('-o', '--output',
                               help='Sauvegarder le workflow suggéré')
    parser_suggest.set_defaults(func=cmd_suggest)
    
    # Commande: channels
    parser_channels = subparsers.add_parser('channels',
                                           help='Lister les canaux d\'un fichier')
    parser_channels.add_argument('input', help='Fichier FCS')
    parser_channels.set_defaults(func=cmd_channels)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Exécuter la commande
    args.func(args)


if __name__ == '__main__':
    main()
