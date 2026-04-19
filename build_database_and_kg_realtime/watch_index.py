"""
watch_index.py — Background daemon to automatically detect dataset edits.
Uses `watchdog` to monitor `profiles.csv`. When changes are saved, it
silently incrementally updates the FAISS/BM25 indexes.
"""

import sys
import time
import threading
import pandas as pd
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Ensure Unicode output for terminal logs
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from indexer import HybridIndex

CSV_PATH = "data/profiles.csv"
INDEX_PATH = Path("data")
DEBOUNCE_SECONDS = 1.0

class DatasetChangeHandler(FileSystemEventHandler):
    def __init__(self, idx: HybridIndex):
        super().__init__()
        self.idx = idx
        self._timer = None

    def on_any_event(self, event):
        try:
            if event.is_directory:
                return
            
            # For move events (atomic saves), check dest_path as well
            path1 = getattr(event, 'dest_path', '')
            path2 = getattr(event, 'src_path', '')
            
            p1_name = Path(path1).name if path1 else ''
            p2_name = Path(path2).name if path2 else ''
            
            if p1_name == CSV_PATH or p2_name == CSV_PATH:
                if self._timer is not None:
                    self._timer.cancel()
                self._timer = threading.Timer(DEBOUNCE_SECONDS, self._process_update)
                self._timer.start()
        except Exception as e:
            print(f"[Watchdog ERROR inside event] {e}")

    def _process_update(self):
        print(f"\n[Watchdog] Detected change in '{CSV_PATH}'!")
        print("[Watchdog] Reading dataset & comparing hashes...")
        
        try:
            # Wait briefly to ensure file locks are released by Excel/Editors
            time.sleep(0.5)
            df = pd.read_csv(CSV_PATH)
            
            result = self.idx.update_from_dataframe(df)
            
            # If no actual changes (e.g. user just hit Ctrl+S with no edits)
            if result.get("new", 0) == 0 and result.get("deleted", 0) == 0 and result.get("changed", 0) == 0:
                print("[Watchdog] Hashes identical. No updates needed.")
                return

            print("[Watchdog] Applying updates to index...")
            self.idx.save(INDEX_PATH)
            
            print("="*50)
            print("  [✓] AUTO-UPDATE COMPLETE")
            print(f"      Added:    {result.get('new', 0)}")
            print(f"      Deleted:  {result.get('deleted', 0)}")
            print(f"      Modified: {result.get('changed', 0)}")
            print("="*50)
            
        except Exception as e:
            print(f"\n[Watchdog ERROR] Failed to process update: {e}")

def main():
    if not INDEX_PATH.exists():
        print(f"  [ERROR] '{INDEX_PATH}' not found. Please run build_index.py first.")
        sys.exit(1)
        
    print("\n" + "="*50)
    print("  Component A — Automated Index Watcher Started")
    print("="*50)
    
    print("[INIT] Loading AI Models & Index into memory...")
    idx = HybridIndex.load(INDEX_PATH)
    print(f"[INIT] Memory loaded! Watching '{CSV_PATH}' for saves...\n")
    print("(Press Ctrl+C to stop watcher)")

    # Set up watchdog
    event_handler = DatasetChangeHandler(idx)
    observer = Observer()
    
    # Watch the current directory with absolute path to fix Windows bugs
    observer.schedule(event_handler, path=str(Path(".").resolve()), recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping Watcher...")
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
