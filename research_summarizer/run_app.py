import traceback
import sys

print("Starting wrapper...")
try:
    import app
    print("App imported successfully! Starting app...")
    app.app.run(host=app.FLASK_HOST, port=app.FLASK_PORT, debug=False)
except Exception as e:
    with open("crash.log", "w") as f:
        f.write("Exception caught:\n")
        f.write(traceback.format_exc())
    print("App crashed. Check crash.log")
    sys.exit(1)
