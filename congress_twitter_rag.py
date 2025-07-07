#!/usr/bin/env python3
"""
Congress Library Twitter Narrative Analysis RAG System
国会图书馆推特叙事分析RAG系统

This system analyzes the narrative themes in Congress Library's Twitter data
using relation extraction and retrieval-augmented generation.

Workflow:
1. Input query about narrative themes
2. Retrieve relevant entities
3. Expand entities to relations  
4. Retrieve original text by relation IDs
5. Generate narrative analysis based on original texts
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Set, Tuple, Any, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
import logging
from datetime import datetime
import re

# Environment configuration
from dotenv import load_dotenv
load_dotenv()

# Embedding and similarity search
try:
    from sentence_transformers import SentenceTransformer
    import torch
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Using basic keyword matching.")

# Spark API for text generation
import hmac
import hashlib
import base64
import time
import json
import requests
from urllib.parse import urlencode


@dataclass
class RelationData:
    """Relation data structure"""
    relation_id: int
    subject: str
    verb: str
    object: str
    source_text_id: str
    sentence_index: int
    original_sentence: str
    narrative: str
    relevance_score: float = 0.0


class SparkAPI:
    """讯飞Spark API客户端"""
    
    def __init__(self):
        # 使用用户提供的完整API密码
        self.http_url = 'https://spark-api-open.xf-yun.com/v2/chat/completions'
        self.api_password = 'iSCHWhoBaxMkDDedacPb:wIGPUGYEvzDvDcJiRDSl'
        
        print(f"API配置 - URL: {self.http_url}")
        print(f"API配置 - 使用完整API密码认证")
        
    def _generate_auth_header(self):
        """Generate authentication header using complete API password"""
        return {
            'Authorization': f'Bearer {self.api_password}',
            'Content-Type': 'application/json'
        }
    
    def generate_text(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> str:
        """Generate text using Spark API"""
        try:
            headers = self._generate_auth_header()
            
            # Spark API format
            payload = {
                "model": "x1",  # 使用正确的模型名称
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            
            print(f"Making API request to: {self.http_url}")
            print(f"Using model: {payload['model']}")
            
            response = requests.post(self.http_url, headers=headers, json=payload, timeout=60)
            
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Response JSON keys: {list(result.keys())}")
                
                # Handle different response formats
                if 'choices' in result and result['choices']:
                    return result['choices'][0].get('message', {}).get('content', '')
                elif 'text' in result:
                    return result['text']
                elif 'output' in result:
                    return result['output']
                else:
                    print(f"Unexpected response format: {result}")
                    return "Error: Unexpected response format"
            else:
                error_text = response.text
                print(f"API request failed: {response.status_code} - {error_text}")
                return f"Error: API request failed ({response.status_code}): {error_text}"
                
        except requests.exceptions.Timeout:
            print("API request timed out")
            return "Error: API request timed out"
        except requests.exceptions.ConnectionError:
            print("Failed to connect to API")
            return "Error: Failed to connect to API"
        except Exception as e:
            print(f"Error calling Spark API: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: API call failed - {str(e)}"


class CongressTwitterRAG:
    """Congress Library Twitter Narrative Analysis RAG System"""
    
    def __init__(self, relatio_output_path: str = "./output"):
        """Initialize the RAG system"""
        self.output_path = relatio_output_path
        self.relations_data: List[RelationData] = []
        self.entity_index: Dict[str, List[int]] = defaultdict(list)
        self.source_index: Dict[str, List[int]] = defaultdict(list)
        self.keyword_index: Dict[str, List[int]] = defaultdict(list)
        
        # Initialize components
        self.spark_api = SparkAPI()
        
        # Initialize embeddings if available
        self.embeddings_model = None
        if EMBEDDINGS_AVAILABLE:
            try:
                model_name = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
                self.embeddings_model = SentenceTransformer(model_name)
                print(f"Loaded embedding model: {model_name}")
            except Exception as e:
                print(f"Failed to load embedding model: {e}")
        
        # Configuration
        self.retrieval_top_k = int(os.getenv('RETRIEVAL_TOP_K', 5))
        self.similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD', 0.5))
        self.max_context_length = int(os.getenv('MAX_CONTEXT_LENGTH', 4000))
        
        # Load data
        self._load_relatio_data()
        self._build_indexes()
        
        print(f"RAG System initialized with {len(self.relations_data)} relations")
    
    def _load_relatio_data(self):
        """Load processed relation data from relatio output"""
        try:
            # Load relations with sources
            relations_file = os.path.join(self.output_path, 'relations_with_sources.json')
            if os.path.exists(relations_file):
                with open(relations_file, 'r', encoding='utf-8') as f:
                    relations_data = json.load(f)
                
                for rel in relations_data:
                    relation = RelationData(
                        relation_id=rel['relation_id'],
                        subject=rel['svo_relation']['subject'],
                        verb=rel['svo_relation']['verb'],
                        object=rel['svo_relation']['object'],
                        source_text_id=rel['source_info']['source_text_id'],
                        sentence_index=rel['source_info']['sentence_index'],
                        original_sentence=rel['source_info']['original_sentence'],
                        narrative=rel['pretty_narrative']
                    )
                    self.relations_data.append(relation)
                
                print(f"Loaded {len(self.relations_data)} relations from {relations_file}")
            else:
                print(f"Relations file not found: {relations_file}")
                
        except Exception as e:
            print(f"Error loading relatio data: {e}")
    
    def _build_indexes(self):
        """Build search indexes for efficient retrieval"""
        print("Building search indexes...")
        
        for i, relation in enumerate(self.relations_data):
            # Entity index
            self.entity_index[relation.subject.lower()].append(i)
            self.entity_index[relation.object.lower()].append(i)
            
            # Source index
            self.source_index[relation.source_text_id].append(i)
            
            # Keyword index (from original sentence)
            words = re.findall(r'\b\w+\b', relation.original_sentence.lower())
            for word in words:
                if len(word) > 3:  # Skip very short words
                    self.keyword_index[word].append(i)
        
        print(f"Built indexes: {len(self.entity_index)} entities, {len(self.source_index)} sources, {len(self.keyword_index)} keywords")
    
    def extract_entities_from_query(self, query: str) -> List[str]:
        """Extract potential entities from the query"""
        query_lower = query.lower()
        found_entities = []
        
        # Direct entity matching
        for entity in self.entity_index.keys():
            if entity in query_lower:
                found_entities.append(entity)
        
        # Keyword-based entity discovery
        query_words = re.findall(r'\b\w+\b', query_lower)
        for word in query_words:
            if word in self.entity_index:
                found_entities.append(word)
        
        return list(set(found_entities))
    
    def retrieve_relations_by_entities(self, entities: List[str]) -> List[int]:
        """Retrieve relation indices by entities"""
        relation_indices = set()
        
        for entity in entities:
            entity_lower = entity.lower()
            if entity_lower in self.entity_index:
                relation_indices.update(self.entity_index[entity_lower])
        
        return list(relation_indices)
    
    def expand_relations(self, initial_indices: List[int], max_expansion: int = 50) -> List[int]:
        """Expand relations by finding related entities and their connections"""
        expanded_indices = set(initial_indices)
        
        # Get all entities from initial relations
        related_entities = set()
        for idx in initial_indices:
            if idx < len(self.relations_data):
                relation = self.relations_data[idx]
                related_entities.add(relation.subject.lower())
                related_entities.add(relation.object.lower())
        
        # Find relations involving these entities
        for entity in related_entities:
            if entity in self.entity_index:
                for rel_idx in self.entity_index[entity]:
                    expanded_indices.add(rel_idx)
                    if len(expanded_indices) >= max_expansion:
                        break
                if len(expanded_indices) >= max_expansion:
                    break
        
        return list(expanded_indices)
    
    def keyword_search(self, query: str) -> List[int]:
        """Search relations by keywords in original sentences"""
        query_words = re.findall(r'\b\w+\b', query.lower())
        relation_scores = defaultdict(float)
        
        for word in query_words:
            if word in self.keyword_index:
                for rel_idx in self.keyword_index[word]:
                    relation_scores[rel_idx] += 1.0
        
        # Sort by relevance score
        sorted_relations = sorted(relation_scores.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, score in sorted_relations]
    
    def calculate_relevance_scores(self, relation_indices: List[int], query: str) -> List[Tuple[int, float]]:
        """Calculate relevance scores for relations based on query"""
        scored_relations = []
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        for idx in relation_indices:
            if idx >= len(self.relations_data):
                continue
                
            relation = self.relations_data[idx]
            score = 0.0
            
            # Keyword overlap in original sentence
            sentence_words = set(re.findall(r'\b\w+\b', relation.original_sentence.lower()))
            overlap = len(query_words.intersection(sentence_words))
            score += overlap * 0.5
            
            # Entity matching
            if any(entity in query_lower for entity in [relation.subject.lower(), relation.object.lower()]):
                score += 1.0
            
            # Verb relevance
            if relation.verb.lower() in query_lower:
                score += 0.8
            
            scored_relations.append((idx, score))
        
        # Sort by score
        scored_relations.sort(key=lambda x: x[1], reverse=True)
        return scored_relations
    
    def retrieve_contexts(self, relation_indices: List[int]) -> Dict[str, Any]:
        """Retrieve and organize contexts for reasoning"""
        contexts = {
            'relations': [],
            'source_documents': defaultdict(list),
            'entities': set(),
            'themes': defaultdict(int),
            'original_texts': []
        }
        
        for idx in relation_indices:
            if idx >= len(self.relations_data):
                continue
                
            relation = self.relations_data[idx]
            
            # Relation info
            contexts['relations'].append({
                'id': relation.relation_id,
                'subject': relation.subject,
                'verb': relation.verb,
                'object': relation.object,
                'narrative': relation.narrative,
                'source_id': relation.source_text_id
            })
            
            # Source document grouping
            contexts['source_documents'][relation.source_text_id].append({
                'relation_id': relation.relation_id,
                'sentence': relation.original_sentence,
                'sentence_index': relation.sentence_index
            })
            
            # Entity collection
            contexts['entities'].add(relation.subject)
            contexts['entities'].add(relation.object)
            
            # Theme analysis (based on verbs)
            contexts['themes'][relation.verb] += 1
            
            # Original texts
            contexts['original_texts'].append({
                'source_id': relation.source_text_id,
                'text': relation.original_sentence,
                'relation': f"{relation.subject} {relation.verb} {relation.object}"
            })
        
        contexts['entities'] = list(contexts['entities'])
        return contexts
    
    def generate_narrative_analysis(self, contexts: Dict[str, Any], query: str) -> str:
        """Generate narrative analysis using Spark API"""
        
        # Prepare context summary
        relations_summary = "\n".join([
            f"- {rel['subject']} {rel['verb']} {rel['object']}: {rel['narrative']}"
            for rel in contexts['relations'][:10]  # Limit to top 10
        ])
        
        # Prepare original texts (limited by context length)
        original_texts = []
        total_length = 0
        max_length = self.max_context_length - 1000  # Reserve space for prompt
        
        for text_info in contexts['original_texts'][:20]:  # Limit number of texts
            text_content = f"[{text_info['source_id']}] {text_info['text']}"
            if total_length + len(text_content) > max_length:
                break
            original_texts.append(text_content)
            total_length += len(text_content)
        
        original_texts_str = "\n".join(original_texts)
        
        # Identify key themes
        top_themes = sorted(contexts['themes'].items(), key=lambda x: x[1], reverse=True)[:5]
        themes_str = ", ".join([f"{theme} ({count})" for theme, count in top_themes])
        
        # Generate analysis prompt
        prompt = f"""
You are analyzing the narrative themes in Congress Library's Twitter content. Based on the retrieved relations and original texts, provide a comprehensive analysis of the narrative patterns.

QUERY: {query}

KEY RELATIONS FOUND:
{relations_summary}

MAIN THEMES: {themes_str}

ORIGINAL TEXT EVIDENCE:
{original_texts_str}

Please provide a detailed narrative analysis that addresses:

1. **Dominant Narrative Themes**: What are the main narrative patterns or themes present in Congress Library's Twitter content related to this query?

2. **Key Relationships**: What are the most important subject-verb-object relationships that shape the narrative?

3. **Messaging Strategy**: How does the Congress Library position itself and its mission through these narratives?

4. **Content Focus**: What topics, events, or concepts receive the most attention?

5. **Audience Engagement**: What narrative techniques are used to engage with the audience?

6. **Overall Narrative Melody**: Summarize the overarching narrative "melody" or consistent themes that emerge from this analysis.

Provide your analysis in clear, structured English. Focus on identifying patterns, themes, and the strategic narrative approach used by Congress Library in their Twitter communications.
"""

        print("Generating narrative analysis...")
        analysis = self.spark_api.generate_text(
            prompt=prompt,
            max_tokens=int(os.getenv('MAX_TOKENS', 2048)),
            temperature=float(os.getenv('TEMPERATURE', 0.7))
        )
        
        return analysis
    
    def analyze_narrative_themes(self, query: str) -> Dict[str, Any]:
        """Complete RAG workflow for narrative theme analysis"""
        print(f"\n🔍 Analyzing narrative themes for query: '{query}'")
        print("=" * 60)
        
        # Step 1: Extract entities from query
        entities = self.extract_entities_from_query(query)
        print(f"📝 Step 1 - Extracted entities: {entities}")
        
        # Step 2: Retrieve relations by entities
        entity_relations = self.retrieve_relations_by_entities(entities)
        print(f"🔗 Step 2 - Found {len(entity_relations)} entity-based relations")
        
        # Step 3: Keyword search for broader coverage
        keyword_relations = self.keyword_search(query)
        print(f"🔎 Step 3 - Found {len(keyword_relations)} keyword-based relations")
        
        # Step 4: Combine and expand relations
        all_relations = list(set(entity_relations + keyword_relations[:20]))  # Limit keyword results
        expanded_relations = self.expand_relations(all_relations, max_expansion=30)
        print(f"📈 Step 4 - Expanded to {len(expanded_relations)} total relations")
        
        # Step 5: Calculate relevance scores and rank
        scored_relations = self.calculate_relevance_scores(expanded_relations, query)
        top_relations = [idx for idx, score in scored_relations[:self.retrieval_top_k]]
        print(f"⭐ Step 5 - Selected top {len(top_relations)} most relevant relations")
        
        # Step 6: Retrieve contexts
        contexts = self.retrieve_contexts(top_relations)
        print(f"📚 Step 6 - Retrieved contexts from {len(contexts['source_documents'])} source documents")
        
        # Step 7: Generate narrative analysis
        analysis = self.generate_narrative_analysis(contexts, query)
        print(f"🤖 Step 7 - Generated narrative analysis")
        
        return {
            'query': query,
            'extracted_entities': entities,
            'total_relations_found': len(expanded_relations),
            'top_relations_used': len(top_relations),
            'source_documents_count': len(contexts['source_documents']),
            'key_themes': dict(contexts['themes']),
            'narrative_analysis': analysis,
            'contexts': contexts
        }


def main():
    """Main function to run the RAG analysis"""
    
    # Initialize the RAG system
    print("🚀 Initializing Congress Library Twitter Narrative RAG System...")
    rag_system = CongressTwitterRAG()
    
    # Example queries for analysis
    example_queries = [
        "What are the main educational themes in Congress Library's Twitter narrative?",
        "How does Congress Library present historical preservation in their social media?",
        "What narrative themes relate to public access and democratic values?",
        "How does Congress Library engage with digital humanities and technology?",
        "What stories does Congress Library tell about American culture and heritage?"
    ]
    
    # Allow user to choose or input custom query
    print("\n📋 Choose an analysis query:")
    for i, query in enumerate(example_queries, 1):
        print(f"{i}. {query}")
    print("6. Enter custom query")
    
    try:
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "6":
            query = input("Enter your custom query: ").strip()
        elif choice in ["1", "2", "3", "4", "5"]:
            query = example_queries[int(choice) - 1]
        else:
            print("Invalid choice, using default query...")
            query = example_queries[0]
        
        # Run the analysis
        result = rag_system.analyze_narrative_themes(query)
        
        # Display results
        print("\n" + "=" * 80)
        print("🎯 NARRATIVE ANALYSIS RESULTS")
        print("=" * 80)
        
        print(f"\n📊 Analysis Summary:")
        print(f"   • Query: {result['query']}")
        print(f"   • Entities found: {len(result['extracted_entities'])}")
        print(f"   • Relations analyzed: {result['total_relations_found']}")
        print(f"   • Top relations used: {result['top_relations_used']}")
        print(f"   • Source documents: {result['source_documents_count']}")
        
        print(f"\n🏷️ Key Themes Detected:")
        for theme, count in sorted(result['key_themes'].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   • {theme}: {count} occurrences")
        
        print(f"\n📝 Narrative Analysis:")
        print("-" * 40)
        print(result['narrative_analysis'])
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"narrative_analysis_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # Create a JSON-serializable version
            save_result = {
                'query': result['query'],
                'timestamp': timestamp,
                'extracted_entities': result['extracted_entities'],
                'total_relations_found': result['total_relations_found'],
                'top_relations_used': result['top_relations_used'],
                'source_documents_count': result['source_documents_count'],
                'key_themes': result['key_themes'],
                'narrative_analysis': result['narrative_analysis']
            }
            json.dump(save_result, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
    except KeyboardInterrupt:
        print("\n\n👋 Analysis interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")


if __name__ == "__main__":
    main()
