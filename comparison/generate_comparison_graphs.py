#!/usr/bin/env python3
"""
Generate comparison graphs for each audio file

This script creates visual comparison graphs for each TTS model output
compared against the reference audio, making it easy to see pitch accent differences.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from urllib.parse import quote


# Configuration
API_URL = "http://127.0.0.1:8000"
VOICE_SPLITS_DIR = Path(__file__).parent.parent / "voice_splits"
SAMPLE_DIR = VOICE_SPLITS_DIR / "sample"
SAMPLE_TEXT_FILE = VOICE_SPLITS_DIR / "sample_text.txt"

# TTS models to evaluate
TTS_MODELS = {
    "FastSpeech2 (Mine)": VOICE_SPLITS_DIR / "mine",
    "ElevenLabs Eiko": VOICE_SPLITS_DIR / "eleven_labs_eiko_wav",
    "Google Translate": VOICE_SPLITS_DIR / "google_translate_wav",
    "Ondoku": VOICE_SPLITS_DIR / "ondoku_wav",
}


class GraphGenerator:
    """Generates comparison graphs using Onsei API"""
    
    def __init__(self, api_url: str = API_URL):
        self.api_url = api_url
        self.sentences = self._load_sentences()
        
    def _load_sentences(self) -> List[str]:
        """Load Japanese sentences from text file"""
        with open(SAMPLE_TEXT_FILE, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    def check_api_health(self) -> bool:
        """Check if the Onsei API is running"""
        try:
            response = requests.get(f"{self.api_url}/", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def generate_comparison_graph(
        self,
        reference_file: Path,
        test_file: Path,
        sentence: str,
        output_file: Path,
        show_all_graphs: bool = True,
    ) -> Tuple[Optional[int], Optional[float], Optional[str]]:
        """
        Generate comparison graph and return score/distance
        
        Returns:
            (score, mean_distance, error_message)
        """
        endpoint = f"{self.api_url}/compare/graph.png"
        
        try:
            with open(reference_file, 'rb') as ref_f, open(test_file, 'rb') as test_f:
                files = {
                    'teacher_audio_file': (reference_file.name, ref_f, 'audio/wav'),
                    'student_audio_file': (test_file.name, test_f, 'audio/wav'),
                }
                data = {
                    'sentence': sentence,
                    'align_audios': 'true',
                    'show_all_graphs': str(show_all_graphs).lower(),
                    'alignment_method': 'phonemes',
                    'fallback_if_no_alignment': 'true',
                }
                
                response = requests.post(endpoint, files=files, data=data, timeout=60)
                
                if response.status_code == 200:
                    # Save the graph
                    with open(output_file, 'wb') as f:
                        f.write(response.content)
                    
                    # Also get the score/distance from the data endpoint
                    endpoint_data = f"{self.api_url}/compare/data"
                    ref_f.seek(0)
                    test_f.seek(0)
                    response_data = requests.post(endpoint_data, files=files, data=data, timeout=60)
                    if response_data.status_code == 200:
                        result = response_data.json()
                        return result.get('score'), result.get('mean_distance'), None
                    return None, None, None
                else:
                    error_detail = response.json().get('detail', 'Unknown error')
                    return None, None, f"API error: {error_detail}"
                    
        except requests.exceptions.Timeout:
            return None, None, "Request timeout"
        except requests.exceptions.RequestException as e:
            return None, None, f"Request failed: {str(e)}"
        except Exception as e:
            return None, None, f"Unexpected error: {str(e)}"
    
    def generate_all_graphs(self, output_dir: Path, show_all_graphs: bool = True) -> Dict:
        """Generate comparison graphs for all models and sentences"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for each sentence
        results = {}
        
        print(f"\n{'='*80}")
        print(f"Generating Comparison Graphs")
        print(f"{'='*80}")
        print(f"Output directory: {output_dir}")
        print(f"Number of sentences: {len(self.sentences)}")
        print(f"Number of models: {len(TTS_MODELS)}")
        print()
        
        for sentence_idx, sentence in enumerate(self.sentences, 1):
            print(f"\n[{sentence_idx}/{len(self.sentences)}] Processing: {sentence[:50]}...")
            
            # Create a safe filename for the sentence
            safe_sentence = "".join(c if c.isalnum() or c in "、。！？" else "_" for c in sentence)
            safe_sentence = safe_sentence[:100]  # Limit length
            
            sentence_dir = output_dir / safe_sentence
            sentence_dir.mkdir(exist_ok=True)
            
            reference_file = SAMPLE_DIR / f"{sentence}.wav"
            
            if not reference_file.exists():
                print(f"  ⚠️  Reference file not found: {sentence}.wav")
                continue
            
            sentence_results = {}
            
            for model_name, model_dir in TTS_MODELS.items():
                test_file = model_dir / f"{sentence}.wav"
                
                if not test_file.exists():
                    print(f"  ⚠️  {model_name}: File not found")
                    sentence_results[model_name] = {
                        'success': False,
                        'error': 'File not found'
                    }
                    continue
                
                # Create output filename
                safe_model_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
                output_file = sentence_dir / f"{safe_model_name}.png"
                
                print(f"  Generating graph for {model_name}...", end=' ', flush=True)
                
                score, mean_distance, error = self.generate_comparison_graph(
                    reference_file,
                    test_file,
                    sentence,
                    output_file,
                    show_all_graphs=show_all_graphs
                )
                
                if error:
                    print(f"❌ Failed: {error}")
                    sentence_results[model_name] = {
                        'success': False,
                        'error': error,
                        'graph_file': None
                    }
                else:
                    print(f"✓ Score: {score}%, Distance: {mean_distance:.3f}")
                    sentence_results[model_name] = {
                        'success': True,
                        'score': score,
                        'mean_distance': mean_distance,
                        'graph_file': output_file.relative_to(output_dir)
                    }
                
                # Small delay to avoid overwhelming the API
                time.sleep(0.5)
            
            results[sentence] = sentence_results
        
        return results
    
    def generate_html_index(self, output_dir: Path, results: Dict):
        """Generate an HTML index page to view all comparison graphs"""
        html_file = output_dir / "index.html"
        
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pitch Accent Comparison Graphs</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        .sentence-section {
            background: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .sentence-title {
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
            margin-bottom: 15px;
            padding: 10px;
            background-color: #f0f0f0;
            border-left: 4px solid #4CAF50;
        }
        .model-comparison {
            margin: 15px 0;
            padding: 15px;
            background-color: #fafafa;
            border-radius: 5px;
        }
        .model-name {
            font-weight: bold;
            color: #555;
            margin-bottom: 10px;
        }
        .model-stats {
            color: #666;
            font-size: 0.9em;
            margin-bottom: 10px;
        }
        .graph-container {
            text-align: center;
            margin: 10px 0;
        }
        .graph-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .error {
            color: #d32f2f;
            font-style: italic;
        }
        .success {
            color: #388e3c;
        }
        .nav {
            background: white;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .nav a {
            color: #4CAF50;
            text-decoration: none;
            margin-right: 15px;
        }
        .nav a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <h1>🎵 Pitch Accent Comparison Graphs</h1>
    <div class="nav">
        <a href="#top">Top</a>
"""
        
        # Add sentence links
        for sentence in results.keys():
            safe_sentence = "".join(c if c.isalnum() or c in "、。！？" else "_" for c in sentence)
            safe_sentence = safe_sentence[:100]
            html_content += f'        <a href="#{safe_sentence}">Sentence {list(results.keys()).index(sentence) + 1}</a>\n'
        
        html_content += """    </div>
    
"""
        
        # Add content for each sentence
        for sentence_idx, (sentence, sentence_results) in enumerate(results.items(), 1):
            safe_sentence = "".join(c if c.isalnum() or c in "、。！？" else "_" for c in sentence)
            safe_sentence = safe_sentence[:100]
            
            html_content += f'    <div class="sentence-section" id="{safe_sentence}">\n'
            html_content += f'        <div class="sentence-title">Sentence {sentence_idx}: {sentence}</div>\n'
            
            # Sort models by score (best first)
            sorted_models = sorted(
                sentence_results.items(),
                key=lambda x: (
                    x[1].get('mean_distance', float('inf')) if x[1].get('success') else float('inf')
                )
            )
            
            for model_name, result in sorted_models:
                html_content += f'        <div class="model-comparison">\n'
                html_content += f'            <div class="model-name">{model_name}</div>\n'
                
                if result.get('success'):
                    score = result.get('score', 'N/A')
                    distance = result.get('mean_distance', 'N/A')
                    graph_file = result.get('graph_file', '')
                    
                    html_content += f'            <div class="model-stats success">\n'
                    html_content += f'                ✓ Score: <strong>{score}%</strong> | '
                    html_content += f'Distance: <strong>{distance:.4f}</strong> (lower is better)\n'
                    html_content += f'            </div>\n'
                    
                    if graph_file:
                        html_content += f'            <div class="graph-container">\n'
                        html_content += f'                <img src="{graph_file}" alt="{model_name} comparison">\n'
                        html_content += f'            </div>\n'
                else:
                    error = result.get('error', 'Unknown error')
                    html_content += f'            <div class="model-stats error">❌ {error}</div>\n'
                
                html_content += f'        </div>\n'
            
            html_content += f'    </div>\n\n'
        
        html_content += """    <div style="margin-top: 40px; padding: 20px; text-align: center; color: #666;">
        <p>Generated by Pitch Accent Comparison Tool</p>
    </div>
</body>
</html>
"""
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n✓ HTML index generated: {html_file}")


def main():
    """Main entry point"""
    print("=" * 80)
    print("Pitch Accent Comparison Graph Generator")
    print("=" * 80)
    print()
    
    # Initialize generator
    generator = GraphGenerator()
    
    # Check API health
    print("Checking Onsei API status...")
    if not generator.check_api_health():
        print("❌ ERROR: Onsei API is not running!")
        print()
        print("Please start the API first:")
        print("  cd comparison/OnseiModified")
        print("  docker build -t onsei .")
        print("  docker build -f Dockerfile.api -t onsei-api .")
        print("  docker run -p 8000:8000 onsei-api")
        print()
        sys.exit(1)
    
    print("✓ Onsei API is running")
    print()
    
    # Create output directory
    output_dir = Path(__file__).parent / "comparison_graphs"
    
    print(f"Output directory: {output_dir}")
    print()
    
    input("Press Enter to start generating graphs (this may take several minutes)...")
    
    # Generate all graphs
    start_time = time.time()
    results = generator.generate_all_graphs(output_dir, show_all_graphs=True)
    elapsed_time = time.time() - start_time
    
    # Generate HTML index
    print("\n" + "=" * 80)
    print("Generating HTML index...")
    print("=" * 80)
    generator.generate_html_index(output_dir, results)
    
    # Summary
    total_graphs = sum(
        sum(1 for r in sentence_results.values() if r.get('success'))
        for sentence_results in results.values()
    )
    
    print()
    print("=" * 80)
    print("Graph Generation Complete!")
    print("=" * 80)
    print(f"Total graphs generated: {total_graphs}")
    print(f"Total time: {elapsed_time:.1f} seconds")
    print(f"Output directory: {output_dir}")
    print(f"HTML index: {output_dir / 'index.html'}")
    print()
    print("Open the HTML file in your browser to view all comparison graphs!")


if __name__ == "__main__":
    main()

