import os

def create_project_structure(base_path):
    directories = [
        'prompts',
        'feedback_loop',
        'reasking_strategy',
        'response_analysis',
        'integration',
        'security',
        'error_handling',
        'models',
        'adapters',
        'ui'
    ]

    files = {
        '': ['main.py', 'setup.py'],
        'prompts': ['__init__.py', 'templates.py'],
        'feedback_loop': ['__init__.py', 'feedback.py'],
        'reasking_strategy': ['__init__.py', 'strategy.py'],
        'response_analysis': ['__init__.py', 'analysis.py'],
        'integration': ['__init__.py', 'external_api.py'],
        'security': ['__init__.py', 'encryption.py'],
        'error_handling': ['__init__.py', 'logger.py'],
        'ui': ['__init__.py', 'interface.py'],
        'models': ['__init__.py', 'models.py'],
        'adapters': ['__init__.py', 'adapters.py'],
    }

    for directory in directories:
        dir_path = os.path.join(base_path, directory)
        os.makedirs(dir_path, exist_ok=True)
        for file in files[directory]:
            open(os.path.join(dir_path, file), 'a').close()

    for file in files['']:
        open(os.path.join(base_path, file), 'a').close()

    print(f"Project structure created at {base_path}")

if __name__ == "__main__":
    base_path = input("Enter the base path for the project: ")
    create_project_structure(base_path)
