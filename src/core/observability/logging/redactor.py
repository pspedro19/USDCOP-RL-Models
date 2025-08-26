"""
Log Redaction Filter
====================
Removes sensitive information from log messages.
"""
import re
import logging
from typing import List

class RedactFilter(logging.Filter):
    """Filter to redact sensitive information from logs."""
    
    def __init__(self):
        super().__init__()
        # Patterns for sensitive data
        self.sensitive_patterns = [
            # MT5 credentials
            r'MT5_LOGIN\s*=\s*[^\s&]+',
            r'MT5_PASSWORD\s*=\s*[^\s&]+',
            r'MT5_SERVER\s*=\s*[^\s&]+',
            
            # API keys and tokens
            r'api_key\s*=\s*[^\s&]+',
            r'api_secret\s*=\s*[^\s&]+',
            r'token\s*=\s*[^\s&]+',
            r'secret\s*=\s*[^\s&]+',
            r'password\s*=\s*[^\s&]+',
            
            # Database credentials
            r'db_password\s*=\s*[^\s&]+',
            r'db_user\s*=\s*[^\s&]+',
            r'redis_password\s*=\s*[^\s&]+',
            
            # URLs with credentials
            r'://[^:]+:[^@]+@',
            
            # Private keys
            r'-----BEGIN PRIVATE KEY-----.*?-----END PRIVATE KEY-----',
            r'-----BEGIN RSA PRIVATE KEY-----.*?-----END RSA PRIVATE KEY-----',
            
            # JWT tokens
            r'eyJ[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*',
        ]
        
        # Compile patterns
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE | re.DOTALL) 
                                 for pattern in self.sensitive_patterns]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log record and redact sensitive information."""
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            record.msg = self._redact_message(record.msg)
        
        # Also check args for sensitive data
        if hasattr(record, 'args') and record.args:
            new_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    new_args.append(self._redact_message(arg))
                else:
                    new_args.append(arg)
            record.args = tuple(new_args)
        
        return True
    
    def _redact_message(self, message: str) -> str:
        """Redact sensitive information from message."""
        redacted = message
        
        for pattern in self.compiled_patterns:
            if pattern.search(redacted):
                if 'MT5_LOGIN' in pattern.pattern:
                    redacted = re.sub(r'MT5_LOGIN\s*=\s*[^\s&]+', 'MT5_LOGIN=REDACTED', redacted, flags=re.IGNORECASE)
                elif 'MT5_PASSWORD' in pattern.pattern:
                    redacted = re.sub(r'MT5_PASSWORD\s*=\s*[^\s&]+', 'MT5_PASSWORD=REDACTED', redacted, flags=re.IGNORECASE)
                elif 'api_key' in pattern.pattern:
                    redacted = re.sub(r'api_key\s*=\s*[^\s&]+', 'api_key=REDACTED', redacted, flags=re.IGNORECASE)
                elif 'api_secret' in pattern.pattern:
                    redacted = re.sub(r'api_secret\s*=\s*[^\s&]+', 'api_secret=REDACTED', redacted, flags=re.IGNORECASE)
                elif 'token' in pattern.pattern:
                    redacted = re.sub(r'token\s*=\s*[^\s&]+', 'token=REDACTED', redacted, flags=re.IGNORECASE)
                elif 'secret' in pattern.pattern:
                    redacted = re.sub(r'secret\s*=\s*[^\s&]+', 'secret=REDACTED', redacted, flags=re.IGNORECASE)
                elif 'password' in pattern.pattern:
                    redacted = re.sub(r'password\s*=\s*[^\s&]+', 'password=REDACTED', redacted, flags=re.IGNORECASE)
                elif '://' in pattern.pattern:
                    redacted = re.sub(r'://[^:]+:[^@]+@', '://REDACTED:REDACTED@', redacted)
                elif 'PRIVATE KEY' in pattern.pattern:
                    redacted = re.sub(r'-----BEGIN PRIVATE KEY-----.*?-----END PRIVATE KEY-----', '[PRIVATE KEY REDACTED]', redacted, flags=re.DOTALL)
                elif 'RSA PRIVATE KEY' in pattern.pattern:
                    redacted = re.sub(r'-----BEGIN RSA PRIVATE KEY-----.*?-----END RSA PRIVATE KEY-----', '[RSA PRIVATE KEY REDACTED]', redacted, flags=re.DOTALL)
                elif 'eyJ' in pattern.pattern:
                    redacted = re.sub(r'eyJ[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*', '[JWT TOKEN REDACTED]', redacted)
        
        return redacted
    
    def add_pattern(self, pattern: str):
        """Add a custom sensitive pattern."""
        self.sensitive_patterns.append(pattern)
        self.compiled_patterns.append(re.compile(pattern, re.IGNORECASE | re.DOTALL))
    
    def remove_pattern(self, pattern: str):
        """Remove a sensitive pattern."""
        try:
            index = self.sensitive_patterns.index(pattern)
            del self.sensitive_patterns[index]
            del self.compiled_patterns[index]
        except ValueError:
            pass
