import sys
import traceback

try:
    from src.models.train_model import main
    print("Successfully imported main function")
    main()
    print("Script executed successfully")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
