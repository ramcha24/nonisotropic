import torch 

def sanity_check(args, resolve_val=None):
    if args is not None:
        for key in args.keys():
            if torch.is_tensor(args[key]) and args[key].isnan().any():
                if resolve_val is not None:
                    try:
                        temp = args[key]
                        temp[temp != temp] = resolve_val
                        args[key] = temp
                    except Exception as e:
                        print(e)
                else:
                    raise ValueError("argument {} has NaN values".format(key))
    return args
