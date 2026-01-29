import argparse
import sys
import time
from src import step0_process_pdf
from src import step1_extract
from src import step2_llm_prep
from src import step3_llm_run
from src import step4_post_process

def main():
    parser = argparse.ArgumentParser(
        description="ChemDataMiner Pipeline: From PDF to Structured Chemical Data",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--step', 
        type=str, 
        default='analysis', 
        choices=['all', 'analysis', '0', '1', '2', '3', '4'],
        help="""Select pipeline step to run:
        0 : [Ingest] Batch process PDFs to Markdown (using Mineru/Magic-PDF) - SLOW!
        1 : [Extract] Extract tables from Markdown files
        2 : [Prep]   Prepare context and prompt for LLM
        3 : [Run]    Run LLM Inference (DeepSeek/OpenAI)
        4 : [Post]   Validate SMILES, desalt, and clean data
        
        analysis : Run steps 1->4 (Default). Assumes PDFs are already processed.
        all      : Run steps 0->4 (Full pipeline).
        """
    )
    
    args = parser.parse_args()

    print(f"🚀 Starting ChemDataMiner Pipeline [Mode: {args.step}]")
    start_time = time.time()

    # --- Step 0: PDF Ingestion (Optional) ---
    if args.step in ['all', '0']:
        print("\n" + "="*50)
        print("STEP 0: PDF Ingestion (Mineru)")
        print("="*50)
        step0_process_pdf.run()

    # --- Step 1: Table Extraction ---
    if args.step in ['all', 'analysis', '1']:
        print("\n" + "="*50)
        print("STEP 1: Extract & Clean Tables")
        print("="*50)
        step1_extract.run()

    # --- Step 2: LLM Prep ---
    if args.step in ['all', 'analysis', '2']:
        print("\n" + "="*50)
        print("STEP 2: Prepare LLM Inputs")
        print("="*50)
        step2_llm_prep.run()

    # --- Step 3: LLM Inference ---
    if args.step in ['all', 'analysis', '3']:
        print("\n" + "="*50)
        print("STEP 3: Run LLM Inference")
        print("="*50)
        step3_llm_run.run()

    # --- Step 4: Post-processing ---
    if args.step in ['all', 'analysis', '4']:
        print("\n" + "="*50)
        print("STEP 4: Post-processing & Validation")
        print("="*50)
        step4_post_process.run()

    elapsed = time.time() - start_time
    print(f"\n Pipeline Finished in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()