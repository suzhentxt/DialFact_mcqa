import json
import random
from typing import List, Dict, Any
import os
import requests
from bs4 import BeautifulSoup
import time
import pickle

class DialFact2MCQAConverter:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.cache_file = "wikipedia_cache.pkl"
        # Initialize url_cache first
        self.url_cache = {}
        # Then load from cache file if exists
        self.load_cache()

    def load_cache(self):
        """Load cached Wikipedia content"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.url_cache = pickle.load(f)
                print(f"Loaded cache with {len(self.url_cache)} URLs")
            except Exception as e:
                print(f"Error loading cache: {e}")
                self.url_cache = {}
        else:
            print("No existing cache found, starting fresh")
            self.url_cache = {}
    
    def save_cache(self):
        """Save cached Wikipedia content"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.url_cache, f)
            print(f"Saved cache with {len(self.url_cache)} URLs")
        except Exception as e:
            print(f"Error saving cache: {e}")
        
    def load_dialfact_data(self, file_path: str) -> List[Dict]:
        """Load DialFact dataset from JSONL file"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def fetch_wikipedia_content(self, url: str) -> str:
        """Fetch full content from Wikipedia URL with caching"""
        # Check cache first
        if url in self.url_cache:
            print(f"Cache hit: {url}")
            return self.url_cache[url]
            
        try:
            print(f"Fetching: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
            
            # Get main content (Wikipedia specific)
            content_div = soup.find('div', {'id': 'mw-content-text'}) or soup.find('div', {'class': 'mw-parser-output'})
            
            if content_div:
                # Extract paragraphs
                paragraphs = content_div.find_all('p')
                content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
                full_content = content[:5000]  # Limit content length
            else:
                # Fallback: get all paragraph text
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
                full_content = content[:5000]
            
            # Cache the result
            self.url_cache[url] = full_content
            return full_content
                
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            # Cache empty result to avoid retry
            self.url_cache[url] = ""
            return ""

    def create_claim_verification_mcqa(self, item: Dict, question_id: int) -> Dict:
        """Create a 3-way claim verification MCQA question with options in question"""
        context_str = " ".join(item["context"])
        
        # Create question with options embedded
        question = f"Given the conversation context, evaluate this claim: '{item['response']}'\n\nWhat is the verification status?\nA. Supports\nB. Refutes\nC. Not Enough Information"
        
        correct_answer = item["response_label"]
        
        # Map correct answer to option letter
        if correct_answer == "SUPPORTS":
            correct_option = "A"
        elif correct_answer == "REFUTES":  
            correct_option = "B"
        else:  # NOT ENOUGH INFO
            correct_option = "C"
        
        # Create search results from evidence with full content
        search_results = []
        if item.get("evidence_list"):
            for i, evidence in enumerate(item["evidence_list"]):
                if len(evidence) >= 3:  # Ensure we have page_name, url, and snippet
                    # Fetch full content from URL
                    full_content = ""
                    if evidence[1]:  # If URL exists
                        # Check if this is a new URL before fetching
                        is_cached = evidence[1] in self.url_cache
                        full_content = self.fetch_wikipedia_content(evidence[1])
                        
                        # Only sleep for new URLs that were actually fetched
                        if not is_cached and full_content:
                            time.sleep(1)  # Rate limiting
                    
                    search_result = {
                        "page_name": evidence[0],  # Wikipedia page title
                        "page_url": evidence[1],   # Wikipedia link  
                        "page_snippet": evidence[2],  # Original snippet
                        "page_result": full_content if full_content else evidence[2],  # Full content or fallback to snippet
                        "page_last_modified": ""   # Not available in DialFact
                    }
                    search_results.append(search_result)
        
        mcqa_item = {
            "id": question_id,
            "question": question,
            "correct_answer": correct_option,
            "options": ["A", "B", "C"],
            "context": context_str,
            "claim": item["response"],
            "search_results": search_results
        }
        
        return mcqa_item
    
    def convert_to_mcqa(self, test_file: str, valid_file: str, output_file: str):
        """Convert DialFact dataset to MCQA format - Claim Verification only"""
        print("Loading DialFact datasets...")
        test_data = self.load_dialfact_data(test_file)
        valid_data = self.load_dialfact_data(valid_file)
        
        # Combine both datasets
        all_data = test_data + valid_data
        print(f"Total items loaded: {len(all_data)}")
        
        mcqa_dataset = {
            "name": "DialFact-MCQA",
            "description": "DialFact dataset converted to Claim Verification MCQA format for evaluating RAG systems",
            "version": "1.0",
            "total_samples": 0,
            "calibration_samples": 5,
            "test_samples": 0,
            "calibration": [],
            "test": []
        }
        
        question_id = 1
        processed_count = 0
        
        # Process each item to create claim verification MCQA questions only
        for i, item in enumerate(all_data):
            if i % 50 == 0:
                print(f"Processing item {i}/{len(all_data)} - Cache size: {len(self.url_cache)}")
            
            # Only process factual claims
            if item.get("type_label") != "factual":
                continue
                
            processed_count += 1
            
            # Create claim verification MCQA
            try:
                claim_mcqa = self.create_claim_verification_mcqa(item, question_id)
                question_id += 1
                
                # Add to calibration set first (limit to first 5 items)
                if len(mcqa_dataset["calibration"]) < 5:
                    mcqa_dataset["calibration"].append(claim_mcqa)
                else:
                    mcqa_dataset["test"].append(claim_mcqa)
                
                # Save cache periodically
                if processed_count % 100 == 0:
                    self.save_cache()
                    
            except Exception as e:
                print(f"Error processing item {i}: {str(e)}")
                continue
        
        # Update counts
        mcqa_dataset["calibration_samples"] = len(mcqa_dataset["calibration"])
        mcqa_dataset["test_samples"] = len(mcqa_dataset["test"])
        mcqa_dataset["total_samples"] = mcqa_dataset["calibration_samples"] + mcqa_dataset["test_samples"]
        
        print(f"Processed {processed_count} factual items from {len(all_data)} total items")
        print(f"Generated {mcqa_dataset['total_samples']} Claim Verification MCQA questions")
        print(f"Calibration: {mcqa_dataset['calibration_samples']}")
        print(f"Test: {mcqa_dataset['test_samples']}")
        
        # Save cache before saving final output
        self.save_cache()
        
        # Save to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(mcqa_dataset, f, indent=2, ensure_ascii=False)
        
        print(f"MCQA dataset saved to {output_file}")
        print(f"Wikipedia cache saved to {self.cache_file}")
        
        return mcqa_dataset

def main():
    converter = DialFact2MCQAConverter()
    
    # Input files
    test_file = "test_split.jsonl"
    valid_file = "valid_split.jsonl" 
    output_file = "dialfact_mcqa.json"
    
    # Check if input files exist
    if not os.path.exists(test_file):
        print(f"Error: {test_file} not found!")
        return
    if not os.path.exists(valid_file):
        print(f"Error: {valid_file} not found!")
        return
    
    # Convert to MCQA format
    mcqa_dataset = converter.convert_to_mcqa(test_file, valid_file, output_file)
    
    # Print sample question
    print("\n=== Sample Claim Verification MCQA ===")
    if mcqa_dataset["calibration"]:
        sample = mcqa_dataset["calibration"][0]
        print(f"ID: {sample['id']}")
        print(f"Question: {sample['question']}")
        print(f"Context: {sample['context'][:200]}...")
        print(f"Claim: {sample['claim']}")
        print(f"Correct Answer: {sample['correct_answer']}")
        if sample['search_results']:
            print(f"Evidence Pages: {len(sample['search_results'])}")
            print(f"First Evidence: {sample['search_results'][0]['page_name']}")
            print(f"Full Content Length: {len(sample['search_results'][0]['page_result'])} chars")

if __name__ == "__main__":
    main()