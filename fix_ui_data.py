import re

with open('server.py', 'r') as f:
    content = f.read()

# Replace the return dict
find_dict = """        return {
            "config": full_config,
            "presets": loaded_presets,
            "initial_gen_result": initial_gen_result_placeholder,
            "model_info": model_info,
            "model_registry": model_registry,
            "available_voices": engine.get_available_voices(),
        }"""

replace_dict = """        sanitized_config = {
            "ui": full_config.get("ui", {}),
            "ui_state": full_config.get("ui_state", {}),
            "generation_defaults": full_config.get("generation_defaults", {}),
            "audio_output": full_config.get("audio_output", {}),
        }
        return {
            "config": sanitized_config,
            "presets": loaded_presets,
            "initial_gen_result": initial_gen_result_placeholder,
            "model_info": model_info,
            "model_registry": model_registry,
            "available_voices": engine.get_available_voices(),
        }"""

content = content.replace(find_dict, replace_dict)

with open('server.py', 'w') as f:
    f.write(content)
