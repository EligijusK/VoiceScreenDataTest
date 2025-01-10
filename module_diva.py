import os
import platform
from time import sleep

headers = ["PVF", "PVS", "AVE", "PVFU", "MI", "MDc", "VL90", "TMAX", "Jit", "Jitc", "JitN", "JitNc"]

def evaluate_audio(path_diva, path_audio):
    if platform.system() != "Windows":
        return dict([(k, "-") for k in headers])

    from pywinauto import Application, ElementNotFoundError

    path_audio = path_audio.replace("/", "\\")  # AMPEX is retarded
    path_ampex = "%s\\test.ampex" % os.path.dirname(path_audio)

    if os.path.exists(path_ampex):
        os.unlink(path_ampex)

    app = Application().start(path_diva)
    win = app["The AMPEX speech analyzer"]
    win.minimize()

    edit_field = win["Edit0"]
    edit_field.set_text(path_audio)

    run_button = win["Run"]
    run_button.click()

    try:
        app["Progress"].minimize()
        while app["Progress"].wait_not("visible", timeout=400):
            pass
    except Exception as e:
        print(e)
        print("timed out")

    app.kill(soft=False)

    sleep(0.01)

    with open(path_ampex, "r") as f:
        values = [float(v) for v in f.readlines()[1].split()[1:]]

    os.unlink(path_ampex)

    return dict(zip(headers, values))


if __name__ == "__main__":
    path_diva = "C:/Users/audry/Desktop/AMPEX_DIVA.exe"
    path_audio = "C:/Users/audry/Desktop/VoiceAnalysis/001.wav"

    evaluate_audio(path_diva, path_audio)
