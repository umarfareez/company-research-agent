import asyncio
import logging
import os
from typing import Any, Dict, List, Union

import google.generativeai as genai

from ..classes import ResearchState

logger = logging.getLogger(__name__)

class Briefing:
    """Creates briefings for each research category and updates the ResearchState."""
    
    def __init__(self) -> None:
        self.max_doc_length = 8000  # Maximum document content length
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        
        # Configure Gemini
        genai.configure(api_key=self.gemini_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')

    async def generate_category_briefing(
        self, docs: Union[Dict[str, Any], List[Dict[str, Any]]], 
        category: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        company = context.get('company', 'Unknown')
        industry = context.get('industry', 'Unknown')
        hq_location = context.get('hq_location', 'Unknown')
        logger.info(f"Generating {category} briefing for {company} using {len(docs)} documents")

        # Send category start status
        if websocket_manager := context.get('websocket_manager'):
            if job_id := context.get('job_id'):
                await websocket_manager.send_status_update(
                    job_id=job_id,
                    status="briefing_start",
                    message=f"Generating {category} briefing",
                    result={
                        "step": "Briefing",
                        "category": category,
                        "total_docs": len(docs)
                    }
                )

        # Enhanced prompts with MongoDB context and specific instructions
        prompts = {
            'company': f"""Create a comprehensive company briefing for {company}, a {industry} company based in {hq_location}.

MongoDB Context: You are analyzing a company that may use databases, data storage, or cloud services. Pay special attention to:
- Technology stack and database solutions
- Data management practices
- Cloud infrastructure choices
- Integration capabilities
- Developer tools and APIs

Key requirements:
1. Start with: "{company} is a [what] that [does what] for [whom]"
2. Structure using these exact headers and bullet points:

### Core Product/Service
* List distinct products/features with technical details
* Include database/data-related capabilities if applicable
* Note any cloud or infrastructure services
* Include only verified technical capabilities

### Technology Stack
* Database technologies used (MongoDB, SQL, NoSQL, etc.)
* Cloud platforms and infrastructure
* Development frameworks and tools
* Data processing and analytics capabilities

### Leadership Team
* List key leadership team members
* Include their roles and technical expertise
* Note any database or cloud technology background

### Target Market
* List specific target audiences and developer communities
* List verified use cases, especially data-driven applications
* List confirmed customers/partners, particularly in tech sector
* Note enterprise vs. developer-focused segments

### Key Differentiators
* List unique technical features
* Database performance and scalability advantages
* Developer experience improvements
* Data security and compliance capabilities

### Business Model
* Discuss product/service pricing, especially SaaS or usage-based models
* List distribution channels (direct, marketplace, partner ecosystem)
* Note freemium, open-source, or enterprise licensing models

3. Each bullet must be a single, complete fact
4. Never mention "no information found" or "no data available"
5. No paragraphs, only bullet points
6. Focus on technical capabilities and data-related features
7. Provide only the briefing. No explanations or commentary.""",

            'industry': f"""Create a focused industry briefing for {company}, a {industry} company based in {hq_location}.

MongoDB Context: Analyze the database and data infrastructure landscape. Consider:
- Database market trends (NoSQL, cloud databases, managed services)
- Developer tool ecosystem
- Cloud transformation patterns
- Data modernization initiatives
- Multi-cloud and hybrid strategies

Key requirements:
1. Structure using these exact headers and bullet points:

### Market Overview
* State {company}'s exact market segment within database/cloud/data infrastructure
* List total addressable market size with year
* List growth rate with year range for database and cloud services
* Note shift from on-premise to cloud-native solutions

### Technology Trends
* NoSQL vs. SQL database adoption patterns
* Cloud-first and multi-cloud strategies
* Serverless and managed database services growth
* Developer experience and DevOps automation trends
* Data analytics and real-time processing demand

### Direct Competition
* List named direct competitors in database/cloud infrastructure space
* List specific competing products and services
* Compare market positions and technology approaches
* Note open-source vs. commercial offerings

### Competitive Advantages
* List unique technical features and performance benefits
* Developer ecosystem and community advantages
* Cloud integration and scalability capabilities
* Security, compliance, and enterprise features

### Market Challenges
* Legacy system migration complexities
* Multi-cloud data consistency challenges
* Developer skill gaps and training needs
* Regulatory compliance requirements
* Cost optimization pressures

2. Each bullet must focus on industry-specific insights
3. No paragraphs, only bullet points
4. Never mention "no information found" or "no data available"
5. Emphasize technology trends and market dynamics
6. Provide only the briefing. No explanation.""",

            'financial': f"""Create a focused financial briefing for {company}, a {industry} company based in {hq_location}.

MongoDB Context: Focus on SaaS, cloud, and database company financial patterns:
- Recurring revenue models (ARR/MRR)
- Usage-based pricing and consumption metrics
- Developer adoption leading to enterprise sales
- Open-source to commercial conversion
- Cloud infrastructure scaling costs

Key requirements:
1. Structure using these headers and bullet points:

### Business Model & Revenue
* Subscription vs. usage-based pricing models
* Freemium to paid conversion strategies
* Developer vs. enterprise customer segments
* Cloud service delivery and scaling economics

### Funding & Investment
* Total funding amount with most recent date
* List each major funding round with date and amount
* List named lead investors and strategic partners
* Note any acquisition offers or strategic investments

### Financial Performance
* Annual Recurring Revenue (ARR) or Monthly Recurring Revenue (MRR) if available
* Customer growth and retention metrics
* Revenue per customer or usage-based metrics
* Geographic revenue distribution

### Market Valuation
* Current or last known valuation with date
* Valuation multiples compared to industry benchmarks
* Public market comparisons if applicable
* Strategic value drivers and market positioning

2. Include specific numbers and dates when possible
3. No paragraphs, only bullet points
4. Never mention "no information found" or "no data available"
5. NEVER repeat the same funding round multiple times
6. Focus on SaaS and cloud-native business metrics
7. NEVER include ranges - use best judgment for exact amounts
8. Provide only the briefing. No explanation or commentary.""",

            'news': f"""Create a focused news briefing for {company}, a {industry} company based in {hq_location}.

MongoDB Context: Prioritize technology and database industry news:
- Product launches and feature announcements
- Cloud platform integrations and partnerships
- Developer community events and open-source initiatives
- Enterprise customer wins and case studies
- Industry conference presentations and thought leadership

Key requirements:
1. Structure into these categories using bullet points:

### Product & Technology Announcements
* New database features and performance improvements
* Cloud service launches and regional expansions
* Developer tools and API enhancements
* Security and compliance certifications

### Strategic Partnerships
* Cloud provider integrations (AWS, Azure, GCP)
* Technology ecosystem partnerships
* Enterprise software integrations
* Open-source community collaborations

### Business Milestones
* Major customer acquisitions and case studies
* Market expansion and international growth
* Funding rounds and strategic investments
* Leadership appointments and advisory additions

### Industry Recognition
* Technology awards and analyst recognition
* Conference keynotes and speaking engagements
* Research reports and market positioning
* Developer community achievements

### Community & Events
* Developer conferences and meetups
* Open-source contributions and releases
* Training programs and certification launches
* Documentation and tutorial improvements

2. Sort newest to oldest within each category
3. One event per bullet point with specific dates when available
4. Focus on technology and business impact
5. Do not mention "no information found" or "no data available"
6. Prioritize database, cloud, and developer-focused news
7. Provide only the briefing. Do not provide explanations or commentary.""",
        }
        
        # Normalize docs to a list of (url, doc) tuples
        items = list(docs.items()) if isinstance(docs, dict) else [
            (doc.get('url', f'doc_{i}'), doc) for i, doc in enumerate(docs)
        ]
        
        # Sort documents by evaluation score (highest first)
        sorted_items = sorted(
            items, 
            key=lambda x: float(x[1].get('evaluation', {}).get('overall_score', '0')), 
            reverse=True
        )
        
        doc_texts = []
        total_length = 0
        for _ , doc in sorted_items:
            title = doc.get('title', '')
            content = doc.get('raw_content') or doc.get('content', '')
            if len(content) > self.max_doc_length:
                content = content[:self.max_doc_length] + "... [content truncated]"
            doc_entry = f"Title: {title}\n\nContent: {content}"
            if total_length + len(doc_entry) < 120000:  # Keep under limit
                doc_texts.append(doc_entry)
                total_length += len(doc_entry)
            else:
                break
        
        separator = "\n" + "-" * 40 + "\n"
        prompt = f"""{prompts.get(category, 'Create a focused, informative and insightful research briefing on the company: {company} in the {industry} industry based on the provided documents.')}

Analyze the following documents and extract key information. Focus on database, cloud, and technology-related content. Provide only the briefing, no explanations or commentary:

{separator}{separator.join(doc_texts)}{separator}

"""
        
        try:
            logger.info("Sending prompt to LLM")
            response = self.gemini_model.generate_content(prompt)
            content = response.text.strip()
            if not content:
                logger.error(f"Empty response from LLM for {category} briefing")
                return {'content': ''}

            # Send completion status
            if websocket_manager := context.get('websocket_manager'):
                if job_id := context.get('job_id'):
                    await websocket_manager.send_status_update(
                        job_id=job_id,
                        status="briefing_complete",
                        message=f"Completed {category} briefing",
                        result={
                            "step": "Briefing",
                            "category": category
                        }
                    )

            return {'content': content}
        except Exception as e:
            logger.error(f"Error generating {category} briefing: {e}")
            return {'content': ''}

    async def create_briefings(self, state: ResearchState) -> ResearchState:
        """Create briefings for all categories in parallel."""
        company = state.get('company', 'Unknown Company')
        websocket_manager = state.get('websocket_manager')
        job_id = state.get('job_id')
        
        # Send initial briefing status
        if websocket_manager and job_id:
            await websocket_manager.send_status_update(
                job_id=job_id,
                status="processing",
                message="Starting research briefings with enhanced MongoDB context",
                result={"step": "Briefing"}
            )

        context = {
            "company": company,
            "industry": state.get('industry', 'Unknown'),
            "hq_location": state.get('hq_location', 'Unknown'),
            "websocket_manager": websocket_manager,
            "job_id": job_id
        }
        logger.info(f"Creating section briefings for {company} with MongoDB context")
        
        # Mapping of curated data fields to briefing categories
        categories = {
            'financial_data': ("financial", "financial_briefing"),
            'news_data': ("news", "news_briefing"),
            'industry_data': ("industry", "industry_briefing"),
            'company_data': ("company", "company_briefing")
        }
        
        briefings = {}

        # Create tasks for parallel processing
        briefing_tasks = []
        for data_field, (cat, briefing_key) in categories.items():
            curated_key = f'curated_{data_field}'
            curated_data = state.get(curated_key, {})
            
            if curated_data:
                logger.info(f"Processing {data_field} with {len(curated_data)} documents")
                
                # Create task for this category
                briefing_tasks.append({
                    'category': cat,
                    'briefing_key': briefing_key,
                    'data_field': data_field,
                    'curated_data': curated_data
                })
            else:
                logger.info(f"No data available for {data_field}")
                state[briefing_key] = ""

        # Process briefings in parallel with rate limiting
        if briefing_tasks:
            # Rate limiting semaphore for LLM API
            briefing_semaphore = asyncio.Semaphore(2)  # Limit to 2 concurrent briefings
            
            async def process_briefing(task: Dict[str, Any]) -> Dict[str, Any]:
                """Process a single briefing with rate limiting."""
                async with briefing_semaphore:
                    result = await self.generate_category_briefing(
                        task['curated_data'],
                        task['category'],
                        context
                    )
                    
                    if result['content']:
                        briefings[task['category']] = result['content']
                        state[task['briefing_key']] = result['content']
                        logger.info(f"Completed {task['data_field']} briefing ({len(result['content'])} characters)")
                    else:
                        logger.error(f"Failed to generate briefing for {task['data_field']}")
                        state[task['briefing_key']] = ""
                    
                    return {
                        'category': task['category'],
                        'success': bool(result['content']),
                        'length': len(result['content']) if result['content'] else 0
                    }

            # Process all briefings in parallel
            results = await asyncio.gather(*[
                process_briefing(task) 
                for task in briefing_tasks
            ])
            
            # Log completion statistics
            successful_briefings = sum(1 for r in results if r['success'])
            total_length = sum(r['length'] for r in results)
            logger.info(f"Generated {successful_briefings}/{len(briefing_tasks)} briefings with total length {total_length}")

        state['briefings'] = briefings
        return state

    async def run(self, state: ResearchState) -> ResearchState:
        return await self.create_briefings(state)
