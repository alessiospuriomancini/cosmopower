from cosmosis.datablock import names
from cosmosis.datablock import option_section
import ast
import sys
import traceback

try:
    import cosmopower as cp
except ImportError:
    sys.stderr.write("Error in CosmoPower. Import failed.\n")

import numpy as np

cosmo = names.cosmological_parameters
cmb_cl = names.cmb_cl


def setup(options):

    config = {
        # 'renames_file': options.get_string(option_section, 'renames_file', default=''),
        'cmb': options.get_bool(option_section, 'cmb', default=True),
        'lmax': options.get_int(option_section, 'lmax', default=-999), # cannot use none for some reason
        'units': options.get_string(option_section, 'units', default='FIRASmuK2'),
        'matter_power_lin': options.get_bool(option_section, 'matter_power_lin', default=False),
        'matter_power_nl': options.get_bool(option_section, 'matter_power_nl', default=False),
        'debug': options.get_bool(option_section, 'debug', default=False),
        'extra_renames': options.get_string(option_section, 'extra_renames', default='{}'),
        'ell_factor': options.get_bool(option_section, 'ell_factor', default=True)
    }

    package_filename = options.get(option_section, 'package_file')

    try:
        parser = cp.YAMLParser(package_filename)
    except FileNotFoundError:
        sys.stderr.write("Error in CosmoPower. package_file {} not found.\n".format(package_filename))

    # if config['renames_file']:
    #     renames = yaml.load(config['renames_file'])

    config['parser'] = parser
    config['networks'] = parser.restore_networks()

    cmb_cl_types = []

    for network in config['networks']:
        cmb_cl_types.append(network.split('/')[-1])

    config['cmb_cl_types'] = cmb_cl_types

    return config


def get_cosmopower_outputs(block, netname, data, config):

    parser = config['parser']

    net_type, cl_type = netname.split('/')

    # Get CMB C_ell
    if net_type=='Cl':
        
        # if not "l" in parser.modes(netname):
        #     raise sys.stderr.write("Error in CosmoPower. \
        #                           Parser file {} does not \
        #                           have l modes defined.".format(parser.yaml_filename))

        block[cmb_cl, 'ell'] = parser.modes(netname).astype(int)

        prefac = np.ones_like(block[cmb_cl, 'ell']).astype(float)
        if parser.settings("Cl/{}".format(cl_type)).get("ell_factor", False):
            prefac /= cp.util.ell_factor(block[cmb_cl, 'ell'], cl_type)
        if config['ell_factor']:
            prefac *= cp.util.ell_factor(block[cmb_cl, 'ell'], cl_type)

        #ToDo: ambiguity of 'Cl' vs 'cmb_cl' here
        block[cmb_cl, cl_type] = data * prefac * cp.util.cmb_unit_factor(cl_type, config['units'], 2.7255)

        if config['lmax'] > 0:
            ell_select = block[cmb_cl, 'ell'] <= config['lmax']
            block[cmb_cl, 'ell'] = block[cmb_cl, 'ell'][ell_select]
            block[cmb_cl, cl_type] = block[cmb_cl, cl_type][ell_select]

    # if config['matter_power_lin']:
    #     block.put_grid("matter_power_lin", "z", z, "k_h", k / h0, "p_k", P.T * h0**3)

    # if config['matter_power_nl']
    #     block.put_grid("matter_power_nl", "z", z, "k_h", k / h0, "p_k", P.T * h0**3)

    if config['matter_power_lin'] is not False:
        raise NotImplemented('Error in CosmoPower. matter_power_lin requested.')

    if config['matter_power_nl'] is not False:
        raise NotImplemented('Error in CosmoPower. matter_power_nl requested.')


def translate_params(p, extra_renames="{}"):

    extra_renames = ast.literal_eval(extra_renames)

    renames = {'omega_c' : 'omega_cdm',
               'tau' : 'tau_reio',
               'h0' : 'h',
               'log1e10as' : 'ln10^{10}A_s',
               'log1e10as' : 'logA',
               'n_s' : 'ns',
               **extra_renames}

    if p in renames.keys():
        return renames[p]
    else:
        return p

def execute(block, config):

    try:

        # get list of cosmological parameter names known about by cosmosis
        params = [p[1] for p in block.keys() if p[0] == cosmo]

        # parameter values under both cosmosis-known and network-known names
        input_params = {
            **{ p : [block[cosmo, p]] for p in params },
            **{ translate_params(p, extra_renames=config['extra_renames']) : [block[cosmo, p]] for p in params }
        }

        for netname in config['networks']:
            
            network = config['networks'][netname]

            # set input parameters
            try:
                used_params = { p : input_params[p] for p in network.parameters }
            except KeyError as err_par:
                print('CosmoPower network parameter {} is not recognised by cosmosis.'.format(err_par))
                print('Please provide a valid rename to this module and/or relation to the consistency module.')
                if 'logA' in err_par.args:
                    print('For logA instead specify log1e10as in the values file.')
                return 1

            # run calculations
            if config['parser'].is_log(netname):
                data = network.ten_to_predictions_np(used_params)[0,:]
            else:
                data = network.predictions_np(used_params)[0,:]

            # extract outputs
            get_cosmopower_outputs(block, netname, data, config)
    except:

        if config['debug']:
            sys.stderr.write("Error in CosmoPower. You set debug=T so here is more debug info:\n")
            traceback.print_exc(file=sys.stderr)
        else:
            sys.stderr.write("Error in CosmoPower. Set debug=T for info: {}\n".format(error))
        return 1
    #finally:
        # Reset for re-use next time

    return 0

def cleanup(config):

    return 0
