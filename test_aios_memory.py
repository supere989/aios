#!/usr/bin/env python3
"""
Test script for AIOS memory management and LLM feedback system
"""

import logging
import time
from ai_model import AIOSController

def test_memory_operations():
    logging.basicConfig(level=logging.INFO)
    controller = AIOSController()
    
    # Test sequence of memory operations
    operations = [
        ("alloc", [128]),           # Allocate 128 bytes
        ("meminfo", []),            # Check memory status
        ("alloc", [256]),           # Allocate another block
        ("meminfo", []),            # Check memory status
        ("memcopy", [0x1000, 0x1200]), # Copy from first to second block
        ("dealloc", [0x1000]),      # Free first block
        ("meminfo", []),            # Final memory status
    ]
    
    results = []
    for op_type, params in operations:
        logging.info(f"\nExecuting: {op_type} {params}")
        
        # Execute operation and get detailed feedback
        cmd, response, train_resp, env_resp, train_resp_2, reward = \
            controller.train_step(op_type, params)
        
        results.append({
            "operation": op_type,
            "parameters": params,
            "command": cmd,
            "response": response,
            "training_feedback": train_resp,
            "environment_feedback": env_resp,
            "final_feedback": train_resp_2,
            "reward": reward
        })
        
        # Allow time for QEMU to process
        time.sleep(1)
        
        # Log results
        logging.info(f"Command: {cmd}")
        logging.info(f"Response: {response}")
        logging.info(f"Reward: {reward}")
        
    return results

if __name__ == "__main__":
    try:
        results = test_memory_operations()
        
        # Print summary
        print("\nTest Summary:")
        for result in results:
            print(f"\nOperation: {result['operation']}")
            print(f"Parameters: {result['parameters']}")
            print(f"Command: {result['command']}")
            print(f"Response: {result['response']}")
            print(f"Reward: {result['reward']}")
            
    except Exception as e:
        logging.error(f"Test failed: {str(e)}")
        raise

