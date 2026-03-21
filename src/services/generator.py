import os
from typing import List, Optional, Dict, Any
import dashscope
from dashscope import Generation

from src.config import config
from src.models import Source


class Generator:
    """LLM answer generation using Qwen via DashScope or Ollama"""

    def __init__(self):
        self.api_key = config.DASHSCOPE_API_KEY
        self.model = config.LLM_MODEL
        self.use_ollama = config.USE_OLLAMA
        self.ollama_base_url = config.OLLAMA_BASE_URL
        self.ollama_model = config.OLLAMA_MODEL

    def _build_prompt(self, question: str, sources: List[Source]) -> str:
        """Build prompt with context from retrieved sources"""
        context_parts = []
        for i, source in enumerate(sources, 1):
            context_parts.append(
                f"[{i}] {source.file_name} (Page {source.page_label}):\n{source.content}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""你是一个专业的文档问答助手。请根据以下参考文档回答问题。

## 参考文档:
{context}

## 问题: {question}

请基于参考文档给出准确、详细的回答。如果参考文档中没有相关信息，请如实说明。
"""
        return prompt

    def _generate_with_dashscope(self, prompt: str) -> str:
        """Generate answer using DashScope API"""
        if not self.api_key:
            return "错误：未配置 DASHSCOPE_API_KEY"

        try:
            dashscope.api_key = self.api_key

            response = Generation.call(
                model=self.model,
                prompt=prompt,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.8,
            )

            if response.status_code == 200:
                return response.output.get("text", "").strip()
            else:
                return f"生成失败: {response.message}"

        except Exception as e:
            return f"生成答案时出错: {str(e)}"

    def _generate_with_ollama(self, prompt: str) -> str:
        """Generate answer using Ollama local model"""
        import urllib.request
        import urllib.error
        import json

        if not self.ollama_model:
            return "错误：未配置 OLLAMA_MODEL"

        url = f"{self.ollama_base_url}/api/generate"

        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.8,
            }
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result.get("response", "").strip()

        except urllib.error.URLError as e:
            return f"Ollama连接失败: {str(e)}。请确保Ollama服务正在运行。"
        except Exception as e:
            return f"Ollama生成答案时出错: {str(e)}"

    def generate(
        self,
        question: str,
        sources: List[Source]
    ) -> str:
        """
        Generate answer using Qwen LLM (DashScope or Ollama).

        Args:
            question: The user's question
            sources: Retrieved context sources

        Returns:
            Generated answer text
        """
        if not sources:
            return "抱歉，没有找到相关的参考文档来回答您的问题。"

        prompt = self._build_prompt(question, sources)

        if self.use_ollama:
            return self._generate_with_ollama(prompt)
        else:
            return self._generate_with_dashscope(prompt)

    def generate_with_sources(
        self,
        question: str,
        sources: List[Source]
    ) -> Dict[str, Any]:
        """Generate answer and include source information"""
        answer = self.generate(question, sources)

        return {
            "answer": answer,
            "sources": sources,
            "question": question,
            "model": self.ollama_model if self.use_ollama else self.model,
        }


generator = Generator()