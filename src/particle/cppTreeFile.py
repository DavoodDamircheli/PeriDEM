import re

def list_classes_methods_functions(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    inside_class = False
    inside_template = False
    for line in lines:
        line = line.strip()

        # Find template declarations
        if re.match(r'template\s*<', line):
            inside_template = True

        # Find class names
        class_match = re.match(r'class (\w+)', line)
        if class_match:
            class_name = f"Class: {class_match.group(1)}"
            if inside_template:
                class_name += " (template)"
                inside_template = False  # Reset template flag
            print(class_name)
            inside_class = True
            continue

        # Find method names within a class
        if inside_class:
            method_match = re.match(r'(\w+ \w+)\(', line)
            if method_match:
                method_name = f"    Method: {method_match.group(1)}"
                if inside_template:
                    method_name += " (template)"
                    inside_template = False  # Reset template flag
                print(method_name)
            
            if line == "};":
                inside_class = False
                continue

        # Find top-level function names
        function_match = re.match(r'(\w+ \w+)\(', line)
        if function_match and not inside_class:
            function_name = f"Top-level function: {function_match.group(1)}"
            if inside_template:
                function_name += " (template)"
                inside_template = False  # Reset template flag
            print(function_name)

if __name__ == "__main__":
    filename = 'timeloop.h'  # Replace with the actual file name
    list_classes_methods_functions(filename)

