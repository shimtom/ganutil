
def save_architecture(dfilepath, gfilepath, discriminator, generator):
    for p, m in zip((dfilepath, gfilepath), (discriminator, generator)):
        with open(p, 'w') as f:
            extension = p.split('.')[-1]
            if extension == 'yml':
                f.write(m.to_yaml(indent=4))
            elif extension == 'json':
                f.write(m.to_json(indent=4))
            else:
                raise ValueError('Unknown file extension: ' + str(extension))
