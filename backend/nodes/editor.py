import logging
import os
from typing import Any, Dict

from langchain_core.messages import AIMessage
from openai import AsyncOpenAI

from ..classes import ResearchState
from ..utils.references import format_references_section

logger = logging.getLogger(__name__)


class Editor:
    """Compiles individual section briefings into a cohesive final report."""
    
    def __init__(self) -> None:
        self.openai_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        # Configure OpenAI
        self.openai_client = AsyncOpenAI(api_key=self.openai_key)
        
        # Initialize context dictionary for use across methods
        self.context = {
            "company": "Unknown Company",
            "industry": "Unknown",
            "hq_location": "Unknown"
        }

    async def compile_briefings(self, state: ResearchState) -> ResearchState:
        """Compile individual briefing categories from state into a final report."""
        company = state.get('company', 'Unknown Company')
        
        # Update context with values from state
        self.context = {
            "company": company,
            "industry": state.get('industry', 'Unknown'),
            "hq_location": state.get('hq_location', 'Unknown')
        }
        
        # Send initial compilation status
        if websocket_manager := state.get('websocket_manager'):
            if job_id := state.get('job_id'):
                await websocket_manager.send_status_update(
                    job_id=job_id,
                    status="processing",
                    message=f"Starting report compilation for {company}",
                    result={
                        "step": "Editor",
                        "substep": "initialization"
                    }
                )

        context = {
            "company": company,
            "industry": state.get('industry', 'Unknown'),
            "hq_location": state.get('hq_location', 'Unknown')
        }
        
        msg = [f"📑 Compiling final report for {company}..."]
        
        # Pull individual briefings from dedicated state keys
        briefing_keys = {
            'company': 'company_briefing',
            'industry': 'industry_briefing',
            'financial': 'financial_briefing',
            'news': 'news_briefing'
        }

        # Send briefing collection status
        if websocket_manager := state.get('websocket_manager'):
            if job_id := state.get('job_id'):
                await websocket_manager.send_status_update(
                    job_id=job_id,
                    status="processing",
                    message="Collecting section briefings",
                    result={
                        "step": "Editor",
                        "substep": "collecting_briefings"
                    }
                )

        individual_briefings = {}
        for category, key in briefing_keys.items():
            if content := state.get(key):
                individual_briefings[category] = content
                msg.append(f"Found {category} briefing ({len(content)} characters)")
            else:
                msg.append(f"No {category} briefing available")
                logger.error(f"Missing state key: {key}")
        
        if not individual_briefings:
            msg.append("\n⚠️ No briefing sections available to compile")
            logger.error("No briefings found in state")
        else:
            try:
                compiled_report = await self.edit_report(state, individual_briefings, context)
                if not compiled_report or not compiled_report.strip():
                    logger.error("Compiled report is empty!")
                else:
                    logger.info(f"Successfully compiled report with {len(compiled_report)} characters")
            except Exception as e:
                logger.error(f"Error during report compilation: {e}")
        state.setdefault('messages', []).append(AIMessage(content="\n".join(msg)))
        return state
    
    async def edit_report(self, state: ResearchState, briefings: Dict[str, str], context: Dict[str, Any]) -> str:
        """Compile section briefings into a final report and update the state."""
        try:
            company = self.context["company"]
            
            # Step 1: Initial Compilation
            if websocket_manager := state.get('websocket_manager'):
                if job_id := state.get('job_id'):
                    await websocket_manager.send_status_update(
                        job_id=job_id,
                        status="processing",
                        message="Compiling initial research report",
                        result={
                            "step": "Editor",
                            "substep": "compilation"
                        }
                    )

            edited_report = await self.compile_content(state, briefings, company)
            if not edited_report:
                logger.error("Initial compilation failed")
                return ""

            # Step 2: Deduplication and Cleanup
            if websocket_manager := state.get('websocket_manager'):
                if job_id := state.get('job_id'):
                    await websocket_manager.send_status_update(
                        job_id=job_id,
                        status="processing",
                        message="Cleaning up and organizing report",
                        result={
                            "step": "Editor",
                            "substep": "cleanup"
                        }
                    )

            # Step 3: Formatting Final Report
            if websocket_manager := state.get('websocket_manager'):
                if job_id := state.get('job_id'):
                    await websocket_manager.send_status_update(
                        job_id=job_id,
                        status="processing",
                        message="Formatting final report",
                        result={
                            "step": "Editor",
                            "substep": "format"
                        }
                    )
            final_report = await self.content_sweep(state, edited_report, company)
            
            final_report = final_report or ""
            
            logger.info(f"Final report compiled with {len(final_report)} characters")
            if not final_report.strip():
                logger.error("Final report is empty!")
                return ""
            
            logger.info("Final report preview:")
            logger.info(final_report[:500])
            
            # Update state with the final report in two locations
            state['report'] = final_report
            state['status'] = "editor_complete"
            if 'editor' not in state or not isinstance(state['editor'], dict):
                state['editor'] = {}
            state['editor']['report'] = final_report
            logger.info(f"Report length in state: {len(state.get('report', ''))}")
            
            if websocket_manager := state.get('websocket_manager'):
                if job_id := state.get('job_id'):
                    await websocket_manager.send_status_update(
                        job_id=job_id,
                        status="editor_complete",
                        message="Research report completed",
                        result={
                            "step": "Editor",
                            "report": final_report,
                            "company": company,
                            "is_final": True,
                            "status": "completed"
                        }
                    )
            
            return final_report
        except Exception as e:
            logger.error(f"Error in edit_report: {e}")
            return ""
    
    async def compile_content(self, state: ResearchState, briefings: Dict[str, str], company: str) -> str:
        """Initial compilation of research sections."""
        combined_content = "\n\n".join(content for content in briefings.values())
        
        references = state.get('references', [])
        reference_text = ""
        if references:
            logger.info(f"Found {len(references)} references to add during compilation")
            
            # Get pre-processed reference info from curator
            reference_info = state.get('reference_info', {})
            reference_titles = state.get('reference_titles', {})
            
            logger.info(f"Reference info from state: {reference_info}")
            logger.info(f"Reference titles from state: {reference_titles}")
            
            # Use the references module to format the references section
            reference_text = format_references_section(references, reference_info, reference_titles)
            logger.info(f"Added {len(references)} references during compilation")
        
        # Use values from centralized context
        company = self.context["company"]
        industry = self.context["industry"]
        hq_location = self.context["hq_location"]
        
        prompt = f"""You are compiling a comprehensive research report about {company}.

Compiled briefings:
{combined_content}

Create a comprehensive and focused report on {company}, a {industry} company headquartered in {hq_location} that:
1. Integrates information from all sections into a cohesive non-repetitive narrative
2. Maintains important details from each section
3. Logically organizes information and removes transitional commentary / explanations
4. Uses clear section headers and structure

Formatting rules:
Strictly enforce this EXACT document structure:

# {company} Research Report

## Executive Summary
[Summary content with bullet points for key findings]

## Chapter 1: Account Overview
[Company content with ### subsections]

## Chapter 2: Strategic Objectives, Modernization Initiatives and Challenges
[Strategic content with ### subsections]

## Chapter 3: Application Modernization Assessment
[Application assessment content]

## Chapter 4: Data Modelling Assessment
[Data modelling content]

## Chapter 5: Gen AI Stack Assessment
[AI/ML initiatives content]

## Chapter 6: Fraud & Compliance Assessment
[Compliance content]

## Chapter 7: Decision Making Unit (DMU) Analysis
[Leadership and organizational content]

## Chapter 8: DMU Messaging Insights
[Messaging recommendations]

Return the report in clean markdown format. No explanations or commentary."""
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert report editor that compiles research briefings into comprehensive company reports."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                stream=False
            )
            initial_report = response.choices[0].message.content.strip()
            
            # Append the references section after LLM processing
            if reference_text:
                initial_report = f"{initial_report}\n\n{reference_text}"
            
            return initial_report
        except Exception as e:
            logger.error(f"Error in initial compilation: {e}")
            return (combined_content or "").strip()
        
    async def content_sweep(self, state: ResearchState, content: str, company: str) -> str:
        """Sweep the content for any redundant information."""
        # Use values from centralized context
        company = self.context["company"]
        industry = self.context["industry"]
        hq_location = self.context["hq_location"]
        
        prompt = f"""You are an expert briefing editor. You are given a report on {company}.

Current report:
{content}

Transform this report into a MongoDB account intelligence report with the following structure:

# {company} MongoDB Account Intelligence Report

## Executive Summary
- Primary use case opportunity for MongoDB
- Secondary use case opportunities (if applicable)
- Overall account readiness score (1-10)
- Key recommendations
- The 3 Whys summary:
  * What's the immediate database requirement?
  * Why MongoDB?
  * Why Now?

## Chapter 1: Account Overview
- Latest financials (revenue, funding, valuation)
- Business strategic priorities
- Key people moves (joiners/leavers in last 6 months)
- Industry context and competitive landscape

## Chapter 2: Strategic Objectives, Modernization Initiatives and Challenges
- Current modernization initiatives
- Technical challenges and pain points
- Strategic objectives driving technology decisions
- Timeline and urgency factors

## Chapter 3: Application Modernization Assessment
- Relevance score (1-10)
- Specific indicators found
- Current state vs. desired state
- MongoDB opportunity alignment

## Chapter 4: Data Modelling Assessment
- Relevance score (1-10)
- Schema flexibility requirements
- Development velocity needs
- MongoDB opportunity alignment

## Chapter 5: Gen AI Stack Assessment
- Relevance score (1-10)
- AI/ML initiatives identified
- Vector search and RAG requirements
- MongoDB opportunity alignment

## Chapter 6: Fraud & Compliance Assessment
- Relevance score (1-10)
- Compliance requirements
- Real-time processing needs
- MongoDB opportunity alignment

## Chapter 7: Decision Making Unit (DMU) Analysis
- Organizational chart of key decision makers
- LinkedIn activity summary (only if posted in last 0-6 months)
- Key topics each persona is discussing
- Influence mapping

## Chapter 8: DMU Messaging Insights
- Persona-specific messaging recommendations
- Technical depth requirements by role
- Key value propositions to emphasize
- Conversation starters and hooks

## References
[Keep existing references exactly as provided]

Critical rules:
1. Extract and reorganize information from the source report to fit this structure
2. For assessment chapters (3-6), assign relevance scores based on available information
3. If information for a section is not available, note "Information not available" rather than omitting
4. Maintain factual accuracy - do not invent information
5. Use bullet points for clarity
6. Focus on MongoDB-relevant insights and opportunities
7. Keep the references section exactly as provided

Return the transformed report in clean markdown format. No explanations or commentary."""
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4.1-mini", 
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at creating MongoDB account intelligence reports that help sales teams identify opportunities and craft effective messaging strategies."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                stream=True
            )
            
            accumulated_text = ""
            buffer = ""
            
            async for chunk in response:
                if chunk.choices[0].finish_reason == "stop":
                    websocket_manager = state.get('websocket_manager')
                    if websocket_manager and buffer:
                        job_id = state.get('job_id')
                        if job_id:
                            await websocket_manager.send_status_update(
                                job_id=job_id,
                                status="report_chunk",
                                message="Formatting final report",
                                result={
                                    "chunk": buffer,
                                    "step": "Editor"
                                }
                            )
                    break
                    
                chunk_text = chunk.choices[0].delta.content
                if chunk_text:
                    accumulated_text += chunk_text
                    buffer += chunk_text
                    
                    if any(char in buffer for char in ['.', '!', '?', '\n']) and len(buffer) > 10:
                        if websocket_manager := state.get('websocket_manager'):
                            if job_id := state.get('job_id'):
                                await websocket_manager.send_status_update(
                                    job_id=job_id,
                                    status="report_chunk",
                                    message="Formatting final report",
                                    result={
                                        "chunk": buffer,
                                        "step": "Editor"
                                    }
                                )
                        buffer = ""
            
            return (accumulated_text or "").strip()
        except Exception as e:
            logger.error(f"Error in formatting: {e}")
            return (content or "").strip()

    async def run(self, state: ResearchState) -> ResearchState:
        state = await self.compile_briefings(state)
        # Ensure the Editor node's output is stored both top-level and under "editor"
        if 'report' in state:
            if 'editor' not in state or not isinstance(state['editor'], dict):
                state['editor'] = {}
            state['editor']['report'] = state['report']
        return state
