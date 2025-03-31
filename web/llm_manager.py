import os
import yaml
from typing import Dict, List, Optional
from openai import OpenAI
# from anthropic import Anthropic
import requests

class LLMManager:
    def __init__(self, config_path: str = 'llm_config.yml'):
        self.config = self._load_config(config_path)
        self.clients = {}
        self._initialize_clients()
    
    def _load_config(self, config_path: str) -> Dict:
        """加载LLM服务配置"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f'配置文件 {config_path} 不存在')
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _initialize_clients(self):
        """初始化各LLM服务客户端"""
        # 初始化Moonshot客户端
        if self.config['moonshot']['enabled']:
            self.clients['moonshot'] = OpenAI(
                api_key=self.config['moonshot']['api_key'],
                base_url=self.config['moonshot']['base_url']
            )
        
        # 初始化OpenAI客户端
        if self.config['openai']['enabled']:
            self.clients['openai'] = OpenAI(
                api_key=self.config['openai']['api_key'],
                base_url=self.config['openai']['base_url']
            )
        
        # 初始化Claude客户端
        # if self.config['anthropic']['enabled']:
            # self.clients['anthropic'] = Anthropic(
                # api_key=self.config['anthropic']['api_key']
            # )
        
        # 初始化文心一言客户端
        if self.config['wendxin']['enabled']:
            self.clients['wendxin'] = {
                'api_key': self.config['wendxin']['api_key'],
                'secret_key': self.config['wendxin']['secret_key']
            }
    
    def get_available_models(self) -> List[Dict]:
        """获取所有可用的模型列表"""
        available_models = []
        for service, config in self.config.items():
            if config.get('enabled', False):
                for model in config.get('models', []):
                    available_models.append({
                        'service': service,
                        'name': model['name'],
                        'description': model['description'],
                        'max_tokens': model['max_tokens']
                    })
        return available_models
    
    def chat_completion(
        self,
        service: str,
        model: str,
        messages: List[Dict],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> str:
        """统一的聊天补全接口"""
        if service not in self.clients:
            raise ValueError(f'服务 {service} 未启用或不存在')
        
        if service in ['moonshot', 'openai']:
            completion = self.clients[service].chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return completion.choices[0].message.content
        
        elif service == 'anthropic':
            messages = [{'role': m['role'], 'content': m['content']} for m in messages]
            completion = self.clients[service].messages.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return completion.content[0].text
        
        elif service == 'wendxin':
            # 获取access token
            token_url = "https://aip.baidubce.com/oauth/2.0/token"
            params = {
                "grant_type": "client_credentials",
                "client_id": self.clients['wendxin']['api_key'],
                "client_secret": self.clients['wendxin']['secret_key']
            }
            response = requests.post(token_url, params=params)
            access_token = response.json().get('access_token')
            
            # 调用文心一言API
            api_url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{model}?access_token={access_token}"
            payload = {
                "messages": messages,
                "temperature": temperature
            }
            headers = {"Content-Type": "application/json"}
            response = requests.post(api_url, headers=headers, json=payload)
            return response.json()['result']
        
        else:
            raise ValueError(f'不支持的服务类型：{service}')
