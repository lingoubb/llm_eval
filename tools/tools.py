def get_var_from_code(code, *var_name):
    loc = {}
    print(globals())
    exec(code, globals(), loc)
    return (loc[var] for var in var_name)


def load_config(fn_name, path, *args):
    with open(path) as f:
        code = f.read()
    fn, = get_var_from_code(code, fn_name)
    return fn(*args)