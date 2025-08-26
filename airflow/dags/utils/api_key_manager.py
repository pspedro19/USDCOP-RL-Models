"""
API Key Manager for TwelveData
================================
Manages multiple API keys with automatic rotation and usage tracking
"""

import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path

class APIKeyManager:
    """
    Manages multiple API keys with intelligent rotation
    
    Features:
    - Track API usage per key
    - Automatic rotation when limits approached
    - Persistent state across runs
    - Optimal key selection
    """
    
    def __init__(self, api_keys: List[str], state_file: str = "/opt/airflow/data_cache/api_state.json"):
        """
        Initialize API Key Manager
        
        Args:
            api_keys: List of available API keys
            state_file: Path to persist API usage state
        """
        self.api_keys = api_keys
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize state
        self.state = self._load_state()
        
        # Initialize keys if not in state
        for key in self.api_keys:
            if key not in self.state:
                self.state[key] = {
                    'credits_used': 0,
                    'last_used': None,
                    'last_reset': datetime.now().isoformat(),
                    'errors': 0,
                    'success_count': 0,
                    'daily_limit': 800
                }
        
        self._save_state()
        
        # Current active key
        self.current_key_index = 0
        self.current_key = self.api_keys[0]
        
    def _load_state(self) -> Dict:
        """Load API usage state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    # Check if any keys need reset (new day)
                    for key, info in state.items():
                        last_reset = datetime.fromisoformat(info.get('last_reset', datetime.now().isoformat()))
                        if datetime.now().date() > last_reset.date():
                            # Reset daily credits
                            info['credits_used'] = 0
                            info['last_reset'] = datetime.now().isoformat()
                            info['errors'] = 0
                            logging.info(f"🔄 Reset daily credits for API key {key[:10]}...")
                    return state
            except Exception as e:
                logging.warning(f"Could not load API state: {e}")
                return {}
        return {}
    
    def _save_state(self):
        """Save API usage state to file"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2, default=str)
        except Exception as e:
            logging.warning(f"Could not save API state: {e}")
    
    def get_best_key(self) -> Tuple[str, Dict]:
        """
        Get the best available API key based on usage
        
        Returns:
            Tuple of (api_key, state_info)
        """
        # Sort keys by available credits (descending)
        available_keys = []
        
        for key in self.api_keys:
            info = self.state[key]
            credits_remaining = info['daily_limit'] - info['credits_used']
            
            # Skip keys with too many errors or exhausted credits
            if info['errors'] > 5 or credits_remaining <= 10:
                continue
                
            available_keys.append((key, credits_remaining, info))
        
        if not available_keys:
            # All keys exhausted, try to find one with least usage
            logging.warning("⚠️ All API keys near limit, using least exhausted")
            key = min(self.api_keys, key=lambda k: self.state[k]['credits_used'])
            return key, self.state[key]
        
        # Sort by available credits
        available_keys.sort(key=lambda x: x[1], reverse=True)
        best_key = available_keys[0][0]
        
        logging.info(f"🔑 Selected API key: {best_key[:10]}... ({available_keys[0][1]} credits remaining)")
        
        self.current_key = best_key
        return best_key, self.state[best_key]
    
    def record_usage(self, api_key: str, credits_used: int = 1, success: bool = True):
        """
        Record API usage for a key
        
        Args:
            api_key: The API key used
            credits_used: Number of credits consumed
            success: Whether the API call was successful
        """
        if api_key in self.state:
            self.state[api_key]['credits_used'] += credits_used
            self.state[api_key]['last_used'] = datetime.now().isoformat()
            
            if success:
                self.state[api_key]['success_count'] += 1
            else:
                self.state[api_key]['errors'] += 1
            
            self._save_state()
            
            # Log usage
            remaining = self.state[api_key]['daily_limit'] - self.state[api_key]['credits_used']
            logging.debug(f"API {api_key[:10]}... used {credits_used} credits, {remaining} remaining")
    
    def rotate_key(self) -> str:
        """
        Force rotation to next available key
        
        Returns:
            New API key
        """
        old_key = self.current_key
        new_key, info = self.get_best_key()
        
        if new_key != old_key:
            logging.info(f"🔄 Rotated from {old_key[:10]}... to {new_key[:10]}...")
        
        return new_key
    
    def should_rotate(self, api_key: str) -> bool:
        """
        Check if we should rotate to a different key
        
        Args:
            api_key: Current API key
            
        Returns:
            True if rotation is recommended
        """
        if api_key not in self.state:
            return True
            
        info = self.state[api_key]
        credits_remaining = info['daily_limit'] - info['credits_used']
        
        # Rotate if:
        # - Less than 50 credits remaining
        # - More than 3 consecutive errors
        # - Used more than 700 credits (approaching limit)
        if credits_remaining < 50 or info['errors'] > 3 or info['credits_used'] > 700:
            return True
            
        return False
    
    def get_usage_summary(self) -> Dict:
        """Get summary of all API keys usage"""
        summary = {
            'total_keys': len(self.api_keys),
            'keys': []
        }
        
        total_credits_used = 0
        total_credits_available = 0
        
        for key in self.api_keys:
            info = self.state[key]
            credits_remaining = info['daily_limit'] - info['credits_used']
            
            summary['keys'].append({
                'key': f"{key[:10]}...{key[-4:]}",
                'credits_used': info['credits_used'],
                'credits_remaining': credits_remaining,
                'errors': info['errors'],
                'success_count': info['success_count'],
                'last_used': info['last_used']
            })
            
            total_credits_used += info['credits_used']
            total_credits_available += credits_remaining
        
        summary['total_credits_used'] = total_credits_used
        summary['total_credits_available'] = total_credits_available
        summary['total_daily_limit'] = len(self.api_keys) * 800
        
        return summary
    
    def reset_all(self):
        """Reset all API key states (use with caution)"""
        for key in self.api_keys:
            self.state[key] = {
                'credits_used': 0,
                'last_used': None,
                'last_reset': datetime.now().isoformat(),
                'errors': 0,
                'success_count': 0,
                'daily_limit': 800
            }
        self._save_state()
        logging.info("🔄 Reset all API key states")