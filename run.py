import os
import sys
import winreg
import subprocess

def get_env_var(name):
    # Try User Environment
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment", 0, winreg.KEY_READ)
        value, _ = winreg.QueryValueEx(key, name)
        winreg.CloseKey(key)
        return value
    except WindowsError:
        pass

    # Try System Environment
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment", 0, winreg.KEY_READ)
        value, _ = winreg.QueryValueEx(key, name)
        winreg.CloseKey(key)
        return value
    except WindowsError:
        pass
    
    return None

def main():
    api_key = get_env_var("ZHIPUAI_API_KEY")
    if api_key:
        print(f"Found ZHIPUAI_API_KEY in registry: {api_key[:5]}...{api_key[-5:]}")
        os.environ["ZHIPUAI_API_KEY"] = api_key
    
    s2_key = get_env_var("S2_API_KEY")
    if s2_key:
        print(f"Found S2_API_KEY in registry: {s2_key[:5]}...{s2_key[-5:]}")
        os.environ["S2_API_KEY"] = s2_key
    
    if not api_key and "ZHIPUAI_API_KEY" not in os.environ:
        print("ZHIPUAI_API_KEY not found in Windows Registry.")
        print("Please ensure the environment variable is set.")
        return

    # Pass all arguments to main.py
    cmd = [sys.executable, "main.py"] + sys.argv[1:]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
