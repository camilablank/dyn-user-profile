#!/usr/bin/env python3
"""
Create a bar chart showing detection accuracies from all summary files.
"""

import json
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_summary_data(folder_path):
    """Load summary data from all summary files."""
    
    data = []
    
    # Find all summary files
    summary_files = list(folder_path.glob("*_summary.*"))
    
    for summary_file in summary_files:
        try:
            if summary_file.suffix == '.csv':
                # Load CSV summary
                with open(summary_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        data.append({
                            'emotion': row.get('emotion', 'unknown'),
                            'accuracy': float(row.get('accuracy_percent', 0)),
                            'correct': int(row.get('correct_detections', 0)),
                            'total': int(row.get('total_comparisons', 0)),
                            'source': summary_file.stem
                        })
            elif summary_file.suffix == '.json':
                # Load JSON summary
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                    
                    # Extract emotion name from filename or description
                    emotion = 'unknown'
                    if 'calm_steady' in summary_file.name:
                        emotion = 'calm_steady'
                    elif 'worried' in summary_file.name:
                        emotion = 'worried'
                    elif 'neg_low' in summary_file.name:
                        emotion = 'neg_low'
                    elif 'neg_high' in summary_file.name:
                        emotion = 'neg_high'
                    elif 'positive_low' in summary_file.name:
                        emotion = 'positive_low'
                    
                    data.append({
                        'emotion': emotion,
                        'accuracy': summary.get('detection_accuracy_percent', 0),
                        'correct': summary.get('n_correct_detections', 0),
                        'total': summary.get('n_comparisons_evaluated', 0),
                        'source': summary_file.stem
                    })
                    
        except Exception as e:
            print(f"Error loading {summary_file}: {e}")
            continue
    
    return data

def create_accuracy_chart(data, output_path):
    """Create a bar chart showing detection accuracies."""
    
    # Sort data by accuracy (descending)
    data.sort(key=lambda x: x['accuracy'], reverse=True)
    
    # Extract data for plotting
    emotions = [item['emotion'] for item in data]
    accuracies = [item['accuracy'] for item in data]
    correct_counts = [item['correct'] for item in data]
    total_counts = [item['total'] for item in data]
    
    # Create figure with larger size
    plt.figure(figsize=(14, 8))
    
    # Create bars
    bars = plt.bar(range(len(emotions)), accuracies, 
                   color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7', '#3A0CA3'],
                   alpha=0.8, edgecolor='black', linewidth=1)
    
    # Customize the plot
    plt.xlabel('Emotion', fontsize=18, fontweight='bold')
    plt.ylabel('Detection Accuracy (%)', fontsize=18, fontweight='bold')
    plt.title('Steering Detection Accuracy by Emotion', fontsize=20, fontweight='bold', pad=20)
    
    # Set x-axis
    plt.xticks(range(len(emotions)), emotions, fontsize=14, rotation=45, ha='right')
    
    # Set y-axis
    plt.ylim(0, 100)
    plt.yticks(np.arange(0, 101, 10), fontsize=14)
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add value labels on bars
    for i, (bar, acc, corr, tot) in enumerate(zip(bars, accuracies, correct_counts, total_counts)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%\n({corr}/{tot})', ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    
    # Add horizontal line at 50% (random baseline)
    plt.axhline(y=50, color='gray', linestyle=':', alpha=0.7, linewidth=2, label='Random Baseline (50%)')
    
    # Add legend
    plt.legend(fontsize=14, loc='upper right')
    
    # Customize grid and spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Accuracy chart saved to: {output_path}")
    
    # Show the plot
    plt.show()

def print_summary_table(data):
    """Print a summary table of the results."""
    
    print("\n" + "="*80)
    print("DETECTION ACCURACY SUMMARY")
    print("="*80)
    print(f"{'Emotion':<15} {'Accuracy':<10} {'Correct':<8} {'Total':<6} {'Source':<25}")
    print("-"*80)
    
    # Sort by accuracy
    data.sort(key=lambda x: x['accuracy'], reverse=True)
    
    for item in data:
        print(f"{item['emotion']:<15} {item['accuracy']:<10.1f}% {item['correct']:<8} {item['total']:<6} {item['source']:<25}")
    
    print("-"*80)
    
    # Calculate overall statistics
    total_correct = sum(item['correct'] for item in data)
    total_comparisons = sum(item['total'] for item in data)
    overall_accuracy = (total_correct / total_comparisons * 100) if total_comparisons > 0 else 0
    
    print(f"{'OVERALL':<15} {overall_accuracy:<10.1f}% {total_correct:<8} {total_comparisons:<6} {'All emotions':<25}")
    print("="*80)

def main():
    """Main function to create the accuracy chart."""
    
    # Paths
    folder_path = Path("new/judge/actual_final")
    output_path = "new/judge/actual_final/detection_accuracy_chart.png"
    
    # Load data
    print("Loading summary data...")
    data = load_summary_data(folder_path)
    
    if not data:
        print("No summary data found!")
        return
    
    print(f"Loaded data for {len(data)} emotions")
    
    # Print summary table
    print_summary_table(data)
    
    # Create the chart
    print("\nCreating accuracy chart...")
    create_accuracy_chart(data, output_path)

if __name__ == "__main__":
    main() 