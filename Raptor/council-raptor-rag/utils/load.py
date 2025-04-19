from dotenv import load_dotenv
import os

def load_env_vars(env_path=r"..\env-vars.env"):
    
    load_dotenv(env_path)
    env_vars = dict(os.environ)

    return env_vars


if __name__ == "__main__":
    print(load_env_vars().keys())