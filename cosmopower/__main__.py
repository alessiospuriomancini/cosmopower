import sys
from importlib import import_module

# If commands[cmd] = ( "module", "func" ), then running
#    cosmopower cmd [args]
# will invoke the function func in module.py with args as its arguments.
commands = {
    "validate" : ("validate", "validate_yaml"),
    "generate" : ("spectra", "generate_spectra"),
    "train" : ("train", "train_networks"),
    "show-training" : ("train", "show_training"),
    "show-validation" : ("validate", "show_validation"),
}

helpmsg = "Command '{0:s}' not known. Try one of:\n\t" + ", ".join(sorted(commands.keys()))

if __name__ == "__main__":
    try:
        cmd = sys.argv[1].lower()
    except IndexError:
        print("Try:\n\tcosmopower <command> -h\nWhere <command> is one of " + ", ".join(sorted(commands.keys())))
        exit()
    else:
        if cmd == "help":
            print("Try:\n\tcosmopower <command> -h\nWhere <command> is one of " + ", ".join(sorted(commands.keys())))
            exit()
        
        module, func = commands.get(cmd, (None, None))
        
        if func is not None:
            getattr(import_module("cosmopower." + module), func)(sys.argv[2:])
        else:
            print(helpmsg.format(cmd))
