#!/usr/bin/env python3
import ai_model
import os
import time
import logging
import sys
import traceback
import argparse
from datetime import datetime, timedelta

class AIOSWorkflowTester:
    def __init__(self, skip_lm_tests=False):
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("aios_test.log")
            ]
        )
        self.logger = logging.getLogger("AIOSWorkflowTester")
        self.logger.info("Initializing AIOS Workflow Tester")
        
        # Test configuration
        self.skip_lm_tests = skip_lm_tests
        if self.skip_lm_tests:
            self.logger.info("LM Studio tests will be skipped (--skip-lm-tests flag)")
        
        # Environment configuration for LM Studio
        os.environ["LM_STUDIO_URL"] = os.environ.get("LM_STUDIO_URL", "http://10.150.1.8:4891/v1/chat/completions")
        os.environ["LM_STUDIO_BACKUP_URL"] = os.environ.get("LM_STUDIO_BACKUP_URL", "")
        os.environ["LM_RETRY_COUNT"] = os.environ.get("LM_RETRY_COUNT", "0")  # Disable retries
        os.environ["LM_TIMEOUT"] = os.environ.get("LM_TIMEOUT", "300")  # 5 minutes timeout for LM Studio
        os.environ["LOG_LEVEL"] = os.environ.get("LOG_LEVEL", "INFO")
        os.environ["LOG_TIMESTAMP"] = os.environ.get("LOG_TIMESTAMP", "true")
        
        # Test timing and results
        self.test_times = {}
        self.test_results = {}
        
        self.controller = None
        self.qemu_comm = None

    def setup(self):
        """Initialize the AIOS Controller and QEMU communication"""
        try:
            self.logger.info("Setting up AIOS Controller...")
            self.controller = ai_model.AIOSController()
            self.logger.info("Setting up QEMU Communication...")
            self.qemu_comm = ai_model.QEMUComm()
            return True
        except Exception as e:
            self.logger.error(f"Setup failed: {str(e)}")
            traceback.print_exc()
            return False

    def test_qemu_connection(self):
        """Test 1: Test QEMU connection with 'info registers' command"""
        self.logger.info("TEST 1: Testing QEMU connection with 'info registers' command (est. time: 1s)")
        start_time = datetime.now()
        try:
            # Send the info registers command
            self.logger.info("Sending 'info registers' command to QEMU...")
            response = self.qemu_comm.send_command("info registers")
            
            # Log the response
            self.logger.info(f"QEMU response received ({len(response)} bytes)")
            self.logger.info("Response excerpt:")
            self.logger.info("\n".join(response.split('\n')[:10]) + "..." if len(response.split('\n')) > 10 else response)
            
            # Verify the response contains register information
            if "EAX" in response or "RAX" in response or "registers" in response.lower():
                elapsed = (datetime.now() - start_time).total_seconds()
                self.test_times["QEMU Connection"] = elapsed
                self.logger.info(f"✅ QEMU connection test PASSED - Register information found in response ({elapsed:.2f}s)")
                return True
            else:
                elapsed = (datetime.now() - start_time).total_seconds()
                self.test_times["QEMU Connection"] = elapsed
                self.logger.warning(f"❌ QEMU connection test FAILED - No register information found in response ({elapsed:.2f}s)")
                return False
        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            self.test_times["QEMU Connection"] = elapsed
            self.logger.error(f"❌ QEMU connection test FAILED with error: {str(e)} ({elapsed:.2f}s)")
            traceback.print_exc()
            return False

    def test_process_intent(self):
        """Test 2: Test the process_intent method with the 'allocate' intent"""
        self.logger.info("TEST 2: Testing process_intent method with 'allocate' intent (est. time: 1-2s)")
        start_time = datetime.now()
        try:
            # Process the allocate intent
            self.logger.info("Processing 'allocate' intent...")
            cmd, response = self.controller.process_intent("allocate", 10, 1000)
            
            # Check if QEMU command was recognized
            if "unknown command" in response:
                self.logger.warning(f"Command '{cmd}' not recognized by QEMU monitor. Attempting alternative approach.")
                # Try to get memory info as a fallback
                mem_info = self.qemu_comm.send_command("info memory")
                self.logger.info(f"Memory info: {mem_info}")
            
            # Log the results
            elapsed = (datetime.now() - start_time).total_seconds()
            self.test_times["Process Intent"] = elapsed
            self.logger.info(f"Time taken: {elapsed:.2f} seconds")
            self.logger.info(f"Generated command: {cmd}")
            self.logger.info(f"QEMU response: {response}")
            
            # Verify the command contains 'alloc'
            if "alloc" in cmd.lower():
                self.logger.info(f"✅ Process intent test PASSED - Generated 'alloc' command correctly ({elapsed:.2f}s)")
                return True
            else:
                self.logger.warning(f"❌ Process intent test FAILED - Did not generate 'alloc' command ({elapsed:.2f}s)")
                return False
        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            self.test_times["Process Intent"] = elapsed
            self.logger.error(f"❌ Process intent test FAILED with error: {str(e)} ({elapsed:.2f}s)")
            traceback.print_exc()
            return False

    def test_lm_studio_basic(self):
        """Test 3: Simple LM Studio connection test with a basic prompt"""
        self.logger.info("TEST 3: Testing LM Studio connection with a simple prompt (est. time: 30-60s)")
        if self.skip_lm_tests:
            self.logger.info("SKIPPED ⏩ LM Studio test (--skip-lm-tests flag enabled)")
            self.test_times["LM Studio Basic"] = 0
            return None  # Return None to indicate skipped (not pass/fail)
            
        start_time = datetime.now()
        try:
            # Send a simple prompt to LM Studio
            prompt = "Hello! Please respond with a brief message to verify the connection is working."
            self.logger.info("Sending basic prompt to LM Studio...")
            
            # Use direct LM Studio query
            lm_response = self.controller.query_lm_studio(prompt, role="training")
            
            # Calculate the time taken
            elapsed = (datetime.now() - start_time).total_seconds()
            self.test_times["LM Studio Basic"] = elapsed
            
            # Log the response
            if lm_response:
                response_excerpt = lm_response[:300] + "..." if len(lm_response) > 300 else lm_response
                self.logger.info(f"LM Studio response received ({len(lm_response)} chars)")
                self.logger.info(f"Response excerpt: {response_excerpt}")
                self.logger.info(f"✅ LM Studio test PASSED - Valid response received ({elapsed:.2f}s)")
                return True
            else:
                self.logger.warning(f"❌ LM Studio test FAILED - Empty response ({elapsed:.2f}s)")
                return False
        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            self.test_times["LM Studio Basic"] = elapsed
            self.logger.error(f"❌ LM Studio test FAILED with error: {str(e)} ({elapsed:.2f}s)")
            traceback.print_exc()
            return False


    def run_all_tests(self):
        """Run all tests and report results"""
        if not self.setup():
            self.logger.error("Setup failed. Cannot proceed with tests.")
            return False
        
        self.logger.info("==================================================")
        self.logger.info("Starting AIOS Workflow Tests")
        self.logger.info("==================================================")
        
        # Run relevant tests
        test_results = {
            "QEMU Connection": self.test_qemu_connection(),
            "Process Intent": self.test_process_intent(),
            "LM Studio Basic": self.test_lm_studio_basic()
        }
        
        # Report results
        self.logger.info("==================================================")
        self.logger.info("AIOS Workflow Test Results")
        self.logger.info("==================================================")
        for test_name, result in test_results.items():
            if result is None:  # Skipped test
                status = "SKIPPED"
                time_str = ""
            else:
                status = "PASSED" if result else "FAILED"
                time_str = f" ({self.test_times.get(test_name, 0):.2f}s)"
                
            self.logger.info(f"{test_name}: {status}{time_str}")
        
        # Overall result - only consider non-skipped tests
        actual_results = {k: v for k, v in test_results.items() if v is not None}
        if all(actual_results.values()):
            total_time = sum(self.test_times.values())
            self.logger.info(f"✅ All tests PASSED (Total time: {total_time:.2f}s)")
            return True
        else:
            failed_tests = [name for name, result in test_results.items() if result is False]
            total_time = sum(self.test_times.values())
            self.logger.warning(f"❌ Some tests FAILED: {', '.join(failed_tests)} (Total time: {total_time:.2f}s)")
            return False


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test the AIOS workflow components")
    parser.add_argument("--skip-lm-tests", action="store_true", 
                        help="Skip LM Studio tests that may take a long time")
    args = parser.parse_args()
    
    # Run the tests
    tester = AIOSWorkflowTester(skip_lm_tests=args.skip_lm_tests)
    success = tester.run_all_tests()
    
    # Return appropriate exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
