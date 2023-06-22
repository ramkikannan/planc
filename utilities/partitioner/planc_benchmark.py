import json
import sys
config_path = sys.argv[1]
f = open(config_path)
config = json.load(f)
f.close()

print(list(config.keys()))


for config_id in list(config.keys()):
    print(config_id)
    c = config[config_id][0]
    print(c)


    for procs in range(len(c['procs'])):
        # do partitioning here, set partition to false in generated scripts ----> assume no
        # overwrites ----> mkdirs to prevent this ----> beware of using lost of memory...
        for algorithm in c['algorithm']:
            for k in c['k']:
                for iterations in c['iterations']:

                    if algorithm == 9:
                        for alpha in c['alpha']:
                            for beta in c['beta']:
                                pass
                                
                    else:
                        pass
