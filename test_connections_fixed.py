import ai_model
import time
import os

def main():
    # Configure logging for debugging
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["LOG_TIMESTAMP"] = "true"
    
    print("Creating AIOS Controller...")
    controller = ai_model.AIOSController()
    
    print("\n--- TESTING QEMU CONNECTION ---")
    try:
        print("Creating QEMUComm instance...")
        qemu = ai_model.QEMUComm()
        print("Sending command to QEMU...")
        qemu_response = qemu.send_command("info status")
        print(f"QEMU response: {qemu_response}")
    except Exception as e:
        print(f"QEMU connection error: {str(e)}")
    
    print("\n--- TESTING LM STUDIO CONNECTION ---")
    try:
        print("Sending request to LM Studio using valid intent...")
        # Use the only valid intent "allocate"
        intent = "allocate"
        size_mb = 1000
        free_mb = 500
        cmd, response = controller.process_intent(intent, size_mb, free_mb)
        print(f"Command: {cmd}")
        print(f"Response: {response}")
        
        # Try the LM Studio connection directly
        print("\n--- TESTING DIRECT LM STUDIO QUERY ---")
        prompt = "How can I optimize memory allocation in an operating system?"
        lm_response = controller.query_lm_studio(prompt, role="training")
        print(f"LM Studio response: {lm_response}")
    except Exception as e:
        print(f"LM Studio connection error: {str(e)}")
    
    print("\nTests completed.")

if __name__ == "__main__":
    main()
