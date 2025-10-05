"""
告警系统
支持多种告警方式：邮件、短信、Webhook、钉钉、企业微信等
"""

import smtplib
import requests
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from abc import ABC, abstractmethod
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.real_time_monitor import AlertInfo as Alert, AlertLevel

class AlertChannel(ABC):
    """告警通道抽象基类"""
    
    @abstractmethod
    def send_alert(self, alert: Alert) -> bool:
        """发送告警"""
        pass
        
    @abstractmethod
    def test_connection(self) -> bool:
        """测试连接"""
        pass

class EmailAlertChannel(AlertChannel):
    """邮件告警通道"""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, 
                 password: str, from_email: str, to_emails: List[str],
                 use_tls: bool = True):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails
        self.use_tls = use_tls
        self.logger = logging.getLogger(__name__)
        
    def send_alert(self, alert: Alert) -> bool:
        """发送邮件告警"""
        try:
            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[{alert.level.value.upper()}] {alert.title}"
            
            # 邮件内容
            body = self._create_email_body(alert)
            msg.attach(MIMEText(body, 'html', 'utf-8'))
            
            # 发送邮件
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
                
            self.logger.info(f"邮件告警发送成功: {alert.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"邮件告警发送失败: {str(e)}")
            return False
            
    def _create_email_body(self, alert: Alert) -> str:
        """创建邮件内容"""
        level_colors = {
            AlertLevel.INFO: '#17a2b8',
            AlertLevel.WARNING: '#ffc107',
            AlertLevel.ERROR: '#dc3545',
            AlertLevel.CRITICAL: '#6f42c1'
        }
        
        color = level_colors.get(alert.level, '#6c757d')
        
        return f"""
        <html>
        <body>
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <div style="background-color: {color}; color: white; padding: 20px; border-radius: 5px 5px 0 0;">
                    <h2 style="margin: 0;">{alert.title}</h2>
                    <p style="margin: 5px 0 0 0;">级别: {alert.level.value.upper()}</p>
                </div>
                <div style="background-color: #f8f9fa; padding: 20px; border: 1px solid #dee2e6; border-top: none; border-radius: 0 0 5px 5px;">
                    <p><strong>时间:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>来源:</strong> {alert.source}</p>
                    <p><strong>消息:</strong></p>
                    <div style="background-color: white; padding: 15px; border-left: 4px solid {color}; margin: 10px 0;">
                        {alert.message}
                    </div>
                    {self._format_alert_data(alert.data) if alert.data else ''}
                </div>
            </div>
        </body>
        </html>
        """
        
    def _format_alert_data(self, data: Dict[str, Any]) -> str:
        """格式化告警数据"""
        if not data:
            return ""
            
        html = "<p><strong>详细信息:</strong></p><ul>"
        for key, value in data.items():
            html += f"<li><strong>{key}:</strong> {value}</li>"
        html += "</ul>"
        return html
        
    def test_connection(self) -> bool:
        """测试邮件连接"""
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.username, self.password)
            return True
        except Exception as e:
            self.logger.error(f"邮件连接测试失败: {str(e)}")
            return False

class WebhookAlertChannel(AlertChannel):
    """Webhook告警通道"""
    
    def __init__(self, webhook_url: str, headers: Dict[str, str] = None,
                 timeout: int = 10):
        self.webhook_url = webhook_url
        self.headers = headers or {'Content-Type': 'application/json'}
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        
    def send_alert(self, alert: Alert) -> bool:
        """发送Webhook告警"""
        try:
            payload = {
                'id': alert.id,
                'level': alert.level.value,
                'title': alert.title,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'source': alert.source,
                'data': alert.data
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                self.logger.info(f"Webhook告警发送成功: {alert.id}")
                return True
            else:
                self.logger.error(f"Webhook告警发送失败: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Webhook告警发送失败: {str(e)}")
            return False
            
    def test_connection(self) -> bool:
        """测试Webhook连接"""
        try:
            test_payload = {
                'test': True,
                'timestamp': datetime.now().isoformat(),
                'message': 'Connection test'
            }
            
            response = requests.post(
                self.webhook_url,
                json=test_payload,
                headers=self.headers,
                timeout=self.timeout
            )
            
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"Webhook连接测试失败: {str(e)}")
            return False

class DingTalkAlertChannel(AlertChannel):
    """钉钉告警通道"""
    
    def __init__(self, webhook_url: str, secret: str = None):
        self.webhook_url = webhook_url
        self.secret = secret
        self.logger = logging.getLogger(__name__)
        
    def send_alert(self, alert: Alert) -> bool:
        """发送钉钉告警"""
        try:
            # 构建消息
            level_emojis = {
                AlertLevel.INFO: '💡',
                AlertLevel.WARNING: '⚠️',
                AlertLevel.ERROR: '❌',
                AlertLevel.CRITICAL: '🚨'
            }
            
            emoji = level_emojis.get(alert.level, '📢')
            
            message = f"{emoji} **{alert.title}**\n\n"
            message += f"**级别:** {alert.level.value.upper()}\n"
            message += f"**时间:** {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
            message += f"**来源:** {alert.source}\n"
            message += f"**消息:** {alert.message}\n"
            
            if alert.data:
                message += "\n**详细信息:**\n"
                for key, value in alert.data.items():
                    message += f"- {key}: {value}\n"
            
            payload = {
                "msgtype": "markdown",
                "markdown": {
                    "title": alert.title,
                    "text": message
                }
            }
            
            # 如果有密钥，需要签名
            if self.secret:
                payload = self._sign_request(payload)
            
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('errcode') == 0:
                    self.logger.info(f"钉钉告警发送成功: {alert.id}")
                    return True
                else:
                    self.logger.error(f"钉钉告警发送失败: {result.get('errmsg')}")
                    return False
            else:
                self.logger.error(f"钉钉告警发送失败: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"钉钉告警发送失败: {str(e)}")
            return False
            
    def _sign_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """签名请求（如果需要）"""
        # 这里可以实现钉钉的签名逻辑
        # 为简化，直接返回原payload
        return payload
        
    def test_connection(self) -> bool:
        """测试钉钉连接"""
        try:
            test_payload = {
                "msgtype": "text",
                "text": {
                    "content": "连接测试"
                }
            }
            
            response = requests.post(self.webhook_url, json=test_payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('errcode') == 0
            return False
            
        except Exception as e:
            self.logger.error(f"钉钉连接测试失败: {str(e)}")
            return False

class WeChatWorkAlertChannel(AlertChannel):
    """企业微信告警通道"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.logger = logging.getLogger(__name__)
        
    def send_alert(self, alert: Alert) -> bool:
        """发送企业微信告警"""
        try:
            # 构建消息
            level_colors = {
                AlertLevel.INFO: 'info',
                AlertLevel.WARNING: 'warning',
                AlertLevel.ERROR: 'warning',
                AlertLevel.CRITICAL: 'warning'
            }
            
            color = level_colors.get(alert.level, 'info')
            
            content = f"**{alert.title}**\n"
            content += f"级别: <font color=\"{color}\">{alert.level.value.upper()}</font>\n"
            content += f"时间: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
            content += f"来源: {alert.source}\n"
            content += f"消息: {alert.message}"
            
            if alert.data:
                content += "\n\n详细信息:\n"
                for key, value in alert.data.items():
                    content += f"- {key}: {value}\n"
            
            payload = {
                "msgtype": "markdown",
                "markdown": {
                    "content": content
                }
            }
            
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('errcode') == 0:
                    self.logger.info(f"企业微信告警发送成功: {alert.id}")
                    return True
                else:
                    self.logger.error(f"企业微信告警发送失败: {result.get('errmsg')}")
                    return False
            else:
                self.logger.error(f"企业微信告警发送失败: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"企业微信告警发送失败: {str(e)}")
            return False
            
    def test_connection(self) -> bool:
        """测试企业微信连接"""
        try:
            test_payload = {
                "msgtype": "text",
                "text": {
                    "content": "连接测试"
                }
            }
            
            response = requests.post(self.webhook_url, json=test_payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('errcode') == 0
            return False
            
        except Exception as e:
            self.logger.error(f"企业微信连接测试失败: {str(e)}")
            return False

class SlackAlertChannel(AlertChannel):
    """Slack告警通道"""
    
    def __init__(self, webhook_url: str, channel: str = None):
        self.webhook_url = webhook_url
        self.channel = channel
        self.logger = logging.getLogger(__name__)
        
    def send_alert(self, alert: Alert) -> bool:
        """发送Slack告警"""
        try:
            # 构建消息
            level_colors = {
                AlertLevel.INFO: '#36a64f',
                AlertLevel.WARNING: '#ff9900',
                AlertLevel.ERROR: '#ff0000',
                AlertLevel.CRITICAL: '#800080'
            }
            
            color = level_colors.get(alert.level, '#36a64f')
            
            attachment = {
                "color": color,
                "title": alert.title,
                "text": alert.message,
                "fields": [
                    {
                        "title": "级别",
                        "value": alert.level.value.upper(),
                        "short": True
                    },
                    {
                        "title": "来源",
                        "value": alert.source,
                        "short": True
                    },
                    {
                        "title": "时间",
                        "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        "short": False
                    }
                ],
                "ts": int(alert.timestamp.timestamp())
            }
            
            # 添加详细信息
            if alert.data:
                for key, value in alert.data.items():
                    attachment["fields"].append({
                        "title": key,
                        "value": str(value),
                        "short": True
                    })
            
            payload = {
                "attachments": [attachment]
            }
            
            if self.channel:
                payload["channel"] = self.channel
            
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            
            if response.status_code == 200:
                self.logger.info(f"Slack告警发送成功: {alert.id}")
                return True
            else:
                self.logger.error(f"Slack告警发送失败: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Slack告警发送失败: {str(e)}")
            return False
            
    def test_connection(self) -> bool:
        """测试Slack连接"""
        try:
            test_payload = {
                "text": "连接测试",
                "channel": self.channel
            }
            
            response = requests.post(self.webhook_url, json=test_payload, timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"Slack连接测试失败: {str(e)}")
            return False

class AlertRouter:
    """告警路由器"""
    
    def __init__(self):
        self.channels = {}
        self.rules = []
        self.logger = logging.getLogger(__name__)
        
    def add_channel(self, name: str, channel: AlertChannel):
        """添加告警通道"""
        self.channels[name] = channel
        self.logger.info(f"添加告警通道: {name}")
        
    def remove_channel(self, name: str):
        """移除告警通道"""
        if name in self.channels:
            del self.channels[name]
            self.logger.info(f"移除告警通道: {name}")
            
    def add_rule(self, condition: callable, channels: List[str]):
        """添加路由规则"""
        self.rules.append({
            'condition': condition,
            'channels': channels
        })
        self.logger.info(f"添加路由规则: {len(channels)} 个通道")
        
    def route_alert(self, alert: Alert) -> Dict[str, bool]:
        """路由告警"""
        results = {}
        
        # 应用路由规则
        target_channels = set()
        for rule in self.rules:
            try:
                if rule['condition'](alert):
                    target_channels.update(rule['channels'])
            except Exception as e:
                self.logger.error(f"路由规则执行失败: {str(e)}")
        
        # 如果没有匹配的规则，发送到所有通道
        if not target_channels:
            target_channels = set(self.channels.keys())
        
        # 发送告警
        for channel_name in target_channels:
            if channel_name in self.channels:
                try:
                    success = self.channels[channel_name].send_alert(alert)
                    results[channel_name] = success
                except Exception as e:
                    self.logger.error(f"通道 {channel_name} 发送失败: {str(e)}")
                    results[channel_name] = False
            else:
                self.logger.warning(f"通道 {channel_name} 不存在")
                results[channel_name] = False
        
        return results
        
    def test_all_channels(self) -> Dict[str, bool]:
        """测试所有通道"""
        results = {}
        for name, channel in self.channels.items():
            try:
                results[name] = channel.test_connection()
            except Exception as e:
                self.logger.error(f"通道 {name} 测试失败: {str(e)}")
                results[name] = False
        return results

# 预定义的路由规则
def critical_alerts_only(alert: Alert) -> bool:
    """仅关键告警"""
    return alert.level == AlertLevel.CRITICAL

def error_and_critical_alerts(alert: Alert) -> bool:
    """错误和关键告警"""
    return alert.level in [AlertLevel.ERROR, AlertLevel.CRITICAL]

def trading_system_alerts(alert: Alert) -> bool:
    """交易系统告警"""
    return 'trading' in alert.source.lower()

def business_hours_only(alert: Alert) -> bool:
    """仅工作时间"""
    hour = alert.timestamp.hour
    return 9 <= hour <= 17

# 配置示例
def create_default_alert_router() -> AlertRouter:
    """创建默认告警路由器"""
    router = AlertRouter()
    
    # 从环境变量读取配置
    email_config = {
        'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
        'smtp_port': int(os.getenv('SMTP_PORT', '587')),
        'username': os.getenv('EMAIL_USERNAME', ''),
        'password': os.getenv('EMAIL_PASSWORD', ''),
        'from_email': os.getenv('FROM_EMAIL', ''),
        'to_emails': os.getenv('TO_EMAILS', '').split(',') if os.getenv('TO_EMAILS') else []
    }
    
    # 添加邮件通道（如果配置了）
    if all([email_config['username'], email_config['password'], email_config['from_email']]):
        email_channel = EmailAlertChannel(**email_config)
        router.add_channel('email', email_channel)
    
    # 添加钉钉通道（如果配置了）
    dingtalk_webhook = os.getenv('DINGTALK_WEBHOOK')
    if dingtalk_webhook:
        dingtalk_channel = DingTalkAlertChannel(dingtalk_webhook)
        router.add_channel('dingtalk', dingtalk_channel)
    
    # 添加企业微信通道（如果配置了）
    wechat_webhook = os.getenv('WECHAT_WEBHOOK')
    if wechat_webhook:
        wechat_channel = WeChatWorkAlertChannel(wechat_webhook)
        router.add_channel('wechat', wechat_channel)
    
    # 添加Slack通道（如果配置了）
    slack_webhook = os.getenv('SLACK_WEBHOOK')
    if slack_webhook:
        slack_channel = SlackAlertChannel(slack_webhook)
        router.add_channel('slack', slack_channel)
    
    # 添加路由规则
    if router.channels:
        # 关键告警发送到所有通道
        router.add_rule(critical_alerts_only, list(router.channels.keys()))
        
        # 错误告警发送到主要通道
        main_channels = ['email', 'dingtalk', 'wechat']
        available_main_channels = [ch for ch in main_channels if ch in router.channels]
        if available_main_channels:
            router.add_rule(error_and_critical_alerts, available_main_channels)
    
    return router

if __name__ == "__main__":
    # 测试告警系统
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试告警
    test_alert = Alert(
        id="test_001",
        level=AlertLevel.WARNING,
        title="测试告警",
        message="这是一个测试告警消息",
        timestamp=datetime.now(),
        source="test_system",
        data={'test_key': 'test_value'}
    )
    
    # 创建路由器
    router = create_default_alert_router()
    
    if router.channels:
        # 测试连接
        print("测试通道连接:")
        test_results = router.test_all_channels()
        for channel, result in test_results.items():
            print(f"  {channel}: {'✓' if result else '✗'}")
        
        # 发送测试告警
        print("\n发送测试告警:")
        send_results = router.route_alert(test_alert)
        for channel, result in send_results.items():
            print(f"  {channel}: {'✓' if result else '✗'}")
    else:
        print("未配置告警通道，请设置环境变量")
        print("支持的环境变量:")
        print("  SMTP_SERVER, SMTP_PORT, EMAIL_USERNAME, EMAIL_PASSWORD, FROM_EMAIL, TO_EMAILS")
        print("  DINGTALK_WEBHOOK, WECHAT_WEBHOOK, SLACK_WEBHOOK")