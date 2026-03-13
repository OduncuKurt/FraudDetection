import sys
import os
from datetime import datetime

class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.file = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.terminal.flush()
        self.file.flush()

def setup_logger(script_name):
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(script_name))[0]
    log_filename = f"results/{base_name}_{timestamp}.txt"
    sys.stdout = TeeLogger(log_filename)
    print(f"Logging output to {log_filename}")
    return log_filename
