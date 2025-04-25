import requests
import json
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for interacting with Ollama API for maintenance analysis."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "mistral:7b"):
        """Initialize Ollama client.
        
        Args:
            base_url (str): Ollama API base URL
            model (str): Model to use for generation
        """
        self.base_url = base_url
        self.model = model
        
    def generate_maintenance_insights(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate maintenance insights using Ollama.
        
        Args:
            data: Dictionary containing maintenance data and metrics
            
        Returns:
            Dictionary containing AI-generated insights
        """
        try:
            prompt = self._create_maintenance_prompt(data)
            response = self._generate(prompt)
            return self._parse_insights(response)
        except Exception as e:
            logger.error(f"Error generating maintenance insights: {str(e)}")
            return self._get_fallback_insights()
    
    def _create_maintenance_prompt(self, data: Dict[str, Any]) -> str:
        """Create a prompt for maintenance analysis."""
        return f"""
You are a maintenance and reliability expert. Analyze the following structured maintenance data and generate actionable insights.

## Equipment Analysis Summary:
- Total Maintenance Cost: ${data.get('total_cost', 0):,.2f}
- Average Cost per Task: ${data.get('avg_cost', 0):,.2f}
- Number of Equipment: {data.get('n_equipment', 0)}
- Total Maintenance Tasks: {data.get('n_tasks', 0)}

## Cluster Information:
{data.get('clusters', [])}

## Detected Anomalies:
{data.get('anomalies', [])}

## Maintenance Patterns:
{data.get('patterns', [])}

---

Please analyze the data and provide:

1. **Cost Optimization Opportunities** – Highlight any high-cost areas and suggest reduction strategies.
2. **Equipment Reliability Concerns** – Identify unreliable equipment or patterns.
3. **Maintenance Scheduling Recommendations** – Propose adjustments to schedules or frequencies.
4. **Risk Factors and Mitigation Strategies** – Point out risks and how to manage them.
5. **Predictive Maintenance Suggestions** – Suggest data-driven maintenance approaches.

**Output Format (strictly use this structure):**

### Insight 1: [Title]
- **Impact Level:** High/Medium/Low
- **Finding:** [Short description of what was detected]
- **Recommendation:** [What action to take]

[Repeat for each relevant insight]

Avoid generic responses. Base your insights strictly on the data provided above.
"""


    def _generate(self, prompt: str) -> str:
        """Make API call to Ollama."""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            logger.error(f"Ollama API error: {str(e)}")
            raise

    def _parse_insights(self, response: str) -> Dict[str, Any]:
        """Parse Ollama response into structured insights."""
        try:
            # Split response into sections
            sections = response.split('\n\n')
            insights = []
            
            for section in sections:
                if not section.strip():
                    continue
                    
                # Try to identify category and content
                lines = section.split('\n')
                if not lines:
                    continue
                    
                category = lines[0].strip().strip(':.').title()
                content = '\n'.join(lines[1:]).strip()
                
                # Determine impact level based on content
                impact = self._determine_impact(content)
                
                insights.append({
                    'category': category,
                    'finding': self._extract_finding(content),
                    'details': self._extract_details(content),
                    'impact': impact,
                    'action': self._extract_action(content)
                })
            
            return {'insights': insights}
        except Exception as e:
            logger.error(f"Error parsing Ollama response: {str(e)}")
            return self._get_fallback_insights()

    def _determine_impact(self, content: str) -> str:
        """Determine impact level based on content."""
        content_lower = content.lower()
        if any(word in content_lower for word in ['critical', 'severe', 'urgent', 'high risk', 'immediate']):
            return 'High'
        elif any(word in content_lower for word in ['moderate', 'medium', 'potential']):
            return 'Medium'
        return 'Low'

    def _extract_finding(self, content: str) -> str:
        """Extract main finding from content."""
        sentences = content.split('.')
        return sentences[0].strip() if sentences else content.strip()

    def _extract_details(self, content: str) -> str:
        """Extract detailed information from content."""
        sentences = content.split('.')
        return '. '.join(sentences[1:3]).strip() if len(sentences) > 1 else ''

    def _extract_action(self, content: str) -> str:
        """Extract recommended action from content."""
        if 'recommend' in content.lower():
            start = content.lower().find('recommend')
            sentence_end = content.find('.', start)
            if sentence_end != -1:
                return content[start:sentence_end].strip()
        return "Review and implement suggested improvements"

    def _get_fallback_insights(self) -> Dict[str, Any]:
        """Provide fallback insights when Ollama fails."""
        return {
            'insights': [{
                'category': 'System Analysis',
                'finding': 'Using fallback analysis mode',
                'details': 'Automated analysis temporarily using baseline metrics',
                'impact': 'Medium',
                'action': 'Review maintenance data manually and retry analysis'
            }]
        }