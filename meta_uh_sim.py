"""This is an extension of patchsim code : A system for doing metapopulation SEIRHUD* models."""
"Authors: Patchsim + leolmoss"

import time
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def read_config(config_file):
    """Read configuration.

    Configuration files contain one key=value pair per line.
    The following is an example of the contents of a config file::

        PatchFile=test_pop.txt
        NetworkFile=test_net.txt
        NetworkType=Static

        ExposureRate=0.65
        InfectionRate=0.67
        RecoveryRate=0.4
        ScalingFactor=1
        
        
        SeedFile=test_seed.txt
        VaxFile=test_vax.txt
        VaxDelay=4
        VaxEfficacy=0.5

        StartDate=1
        Duration=30

        LoadState=False
        SaveState=True
        SaveFile=checkpoint1.npy

        OutputFile=test1.out
        OutputFormat=Whole
        LogFile=test1.log

    Parameters
    ----------
    config_file : str
        Path to the configuration file.

    Returns
    -------
    dict (str -> str)
        The configuration key value pairs.
    """
    config_df = pd.read_csv(config_file, delimiter="=", names=["key", "val"])
    configs = dict(zip(config_df.key, config_df.val))
    #print(configs)
    configs.setdefault("Model", "Mobility")
    return configs


def load_patch(configs):
    """Load the patch file.

    A patch file contains the population size of a patch.
    The file has two space separated columns.
    Following is an example of a patch file::

        A 10000
        B 10000
        C 10000

    Parameters
    ----------
    configs : dict
        The configuration dictionary.
        Must contain the "PatchFile" pointing to location of patch file.

    Returns
    -------
    DataFrame (names=(id, pops), dtypes=(str, int))
        A dataframe containing populations of patches.
    """
    patch_df = pd.read_csv(
        configs["PatchFile"],
        names=["id", "pops"],
        delimiter=" ",
        dtype={"id": str, "pops": int},
    )
    patch_df.sort_values("id", inplace=True)
    #print(f'len patchdf {len(patch_df)}')
    logger.info("Loaded patch attributes")
    return patch_df


def load_param_file(configs):
    """Load the parameter file.

    A parameter file contains one row per patch.
    Each row must have two or more columns.
    Following is an example of a paremter file::

        B 0 0 0.54 0.54 0.54 0.54 0 0 0 0
        A 0.72

    Parameters
    ----------
    configs : dict
        The configuration dictionary.
        Must contain the "ParamFile" pointing to location of parameter file.
    patch_df : DataFrame
        A dataframe containing populations of patches.

    Returns
    -------
    DataFrame
         A dataframe with one column per patch.
         The column names are IDs of the patches.
         Each column contains the "beta" value of the patch over time.
    """
    param_df = pd.read_csv(
        configs["ParamFile"], delimiter=" ", dtype={0: str}, header=None
    )
    param_df = param_df.set_index(0) #se pone la primera columna como la columna de índice
    param_df = param_df.fillna(method="ffill", axis=1) #llena los Nans con la ultima obervacions en el datafrme disponible
    param_df = param_df.T
    #print(param_df)
    return param_df



# def load_params(configs, patch_df):
#     """Load the simulation parameters.

#     Parameters
#     ----------
#     configs : dict
#         The configuration key value pairs.
#     patch_df : DataFrame
#         A dataframe containing populations of patches.

#     Returns
#     -------
#     dict (str -> float or ndarray)
#         A dictionary of model parameters.
#         The "beta" parameter is a ndarray
#         with shape=(NumPatches x NumTimesteps)
#         and dtype=float.
#     """
#     params = {}
#     params["T"] = int(configs["Duration"])

#     beta = float(configs.get("ExposureRate", 0.0))
#     param_df = param_df.set_index(0)
#     param_df = param_df.fillna(method="ffill", axis=1)
#     param_df = param_df.T
#     #print(param_df)
#     return param_df


def load_params(configs, patch_df):
    """Load the simulation parameters.

    Parameters
    ----------
    configs : dict
        The configuration key value pairs.
    patch_df : DataFrame
        A dataframe containing populations of patches.

    Returns
    -------
    dict (str -> float or ndarray)
        A dictionary of model parameters.
        The "beta" parameter is a ndarray
        with shape=(NumPatches x NumTimesteps)
        and dtype=float.
    """
    params = {}
    params["T"] = int(configs["Duration"])

    beta = float(configs.get("ExposureRate", 0.0)) #En simule, este es fijo = 1.1
    params["beta2"] = float(configs.get("ExposureRate2", 0.0)) # En simule este es el que se ajusta
    #print(params["beta2"])
    params["beta"] = np.full((len(patch_df), params["T"]), beta) #matriz (patch,tiempo) llena de beta
    params["alpha"] = float(configs.get("InfectionRate", 0.0))
    #Transitions rates Symptomatics
    params["gamma_ir"] = float(configs.get("TransitionRate_ir", 0.0))#recovery
    params["gamma_ih"] = float(configs.get("TransitionRate_ih", 0.0))
    params["gamma_iu"] = float(configs.get("TransitionRate_iu", 0.0))
    params["gamma_id"] = float(configs.get("TransitionRate_id", 0.0))
    
    #Recovery rates Asymptomatics
    params["gamma_ar"] = float(configs.get("RecoveryRateAsymptomatic", 0.0))

    #Proportions Symptomatic
    params["sigma_ir"] = float(configs.get("Prob_S_ymptomaticToRecover", 1.0))
    params["sigma_ih"] = float(configs.get("Prob_S_ymptomaticToHospital", 0.0))
    params["sigma_iu"] = float(configs.get("Prob_S_ymptomaticToUci", 0.0))
    #Hospitalized 
    params["sigma_hr"] = float(configs.get("Prob_HospitalizedToRecover", 0.0))
    params["sigma_hu"] = float(configs.get("Prob_HospitalizedToUci", 0.0))

    params["gamma_hr"] = float(configs.get("TransitionRate_hr", 0.0))#recovery
    params["gamma_hu"] = float(configs.get("TransitionRate_hu", 0.0))
    params["gamma_hd"] = float(configs.get("TransitionRate_hd", 0.0))
    #Uci
    params["sigma_uh"] = float(configs.get("Prob_UciToHospitals", 0.0))
    params["gamma_uh"] = float(configs.get("TransitionRate_uh", 0.0))#recovery
    params["gamma_ud"] = float(configs.get("TransitionRate_ud", 0.0))

    params["factor_H"]=(1- params["sigma_hr"]*params["gamma_hr"]- params["sigma_hu"]*params["gamma_hu"]-params["gamma_hd"]+params["sigma_hr"]*params["gamma_hd"]+params["sigma_hu"]*params["gamma_hd"])
    params["factor_I"]=(1-params["sigma_ir"]*params["gamma_ir"]-params["sigma_ih"]*params["gamma_ih"]-params["sigma_iu"]*params["gamma_iu"])-params["gamma_id"]*(1-params["sigma_ir"]-params["sigma_ih"]-params["sigma_iu"])
    
    params["factor_U"]=1-params["sigma_uh"]*params["gamma_uh"]-params["gamma_ud"]+params["sigma_uh"]*params["gamma_ud"]
    
    params["sigma_ar"] = float(configs.get("Prob_A_symptomaticToRecover", 1.0))
    
    logger.info(
        "Parameter: alpha=%e, beta=%e, gamma=%e", params["alpha"], beta, params["gamma_ir"]
    )

    if "ParamFile" in configs:
        param_df = load_param_file(configs)
        # print('PARAM_DF')
        # display(param_df)
        for i, id_ in enumerate(patch_df["id"]): #recorre id upzs
            if id_ in param_df.columns:
                xs = param_df[id_]
                params["beta"][i, 0 : len(xs)] = xs
        # print('PARAM BETA')
        # display(pd.DataFrame(params["beta"]))
        # pd.DataFrame(params["beta"]).to_csv('../PARAMS_BETA_11.csv')
        logger.info("Loaded disease parameters from ParamFile")
        
        #print(f'type {type(params["beta"])} shape {params["beta"].shape}')
    else:
        logger.info("No ParamFile loaded")
    
    # np.save('paramsBEta.npy',params["beta2"]* params["beta"])
    
    ### Optional parameters
    params["scaling"] = float(configs.get("ScalingFactor", 1.0))
    params["vaxeff"] = float(configs.get("VaxEfficacy", 1.0))
    params["delta"] = float(configs.get("WaningRate", 0.0))
    params["kappa"] = float(configs.get("AsymptomaticReduction", 1.0))#Our 1-theta
    params["symprob"] = float(configs.get("SymptomaticProbability", 1.0))
    params["epsilon"] = float(configs.get("PresymptomaticReduction", 1.0))

    if params["delta"]:
        logger.info("Found WaningRate. Running SEIRS model.")

    return params


def load_seed(configs, params, patch_df):
    """Load the disease seeding schedule file.

    A seed file contains the disease seeding schedule.
    Following is an example of the contents of a seed file::

        0 A 20
        0 B 20
        1 C 20
        2 C 30

    Parameters
    ----------
    configs : dict
        The configuration dictionary.
    params: dict (str -> float or ndarray)
        A dictionary of model parameters.
    patch_df : DataFrame
        A dataframe containing populations of patches.

    Returns
    -------
    ndarray shape=(NumTimsteps x NumPatches)
        A seeding schedule matrix
    """
    if "SeedFile" not in configs:
        logger.info("Continuing without seeding")
        return np.zeros((params["T"], len(patch_df)))

    seed_df = pd.read_csv(
        configs["SeedFile"],
        delimiter=" ",
        names=["Day", "Id", "Count"],
        dtype={"Id": str},
    )

    seed_mat = np.zeros((params["T"], len(patch_df)))
    seed_df = seed_df[seed_df.Day<params["T"]] ### Skipping seeds after end of simulation
    patch_idx = {id_: i for i, id_ in enumerate(patch_df["id"])}
    for day, id_, count in seed_df.itertuples(index=False, name=None):
        idx = patch_idx[id_]
        seed_mat[day, idx] = count

    logger.info("Loaded seeding schedule")
    return seed_mat


def load_vax(configs, params, patch_df):
    """Load the vaccination schedule file.

    A vax file contains the vaccination schedule.
    Following is an example of the contents of the vax file::

        0 A 10
        2 B 10
        5 C 10

    Parameters
    ----------
    configs : dict
        The configuration dictionary.
    params: dict (str -> float or ndarray)
        A dictionary of model parameters.
    patch_df : DataFrame
        A dataframe containing populations of patches.

    Returns
    -------
    ndarray shape=(NumTimsteps x NumPatches)
        A vaccination schedule matrix (NumTimsteps x NumPatches)
    """
    vax_mat = np.zeros((params["T"], len(patch_df)), dtype=int)

    if "VaxFile" not in configs:
        return vax_mat

    vax_df = pd.read_csv(
        configs["VaxFile"],
        delimiter=" ",
        names=["Day", "Id", "Count"],
        dtype={"Id": str, "Count": int},
    )
    vax_delay = int(configs.get("VaxDelay", 0))

    patch_idx = {id_: i for i, id_ in enumerate(patch_df["id"])}
    for day, id_, count in vax_df.itertuples(index=False, name=None):
        idx = patch_idx[id_]
        day = day + vax_delay
        vax_mat[day, idx] = count

    return vax_mat


def load_Theta(configs, patch_df):
    """Load the patch connectivity network.

    This function loads the dynamic network connectity file.
    The following is an example of the network connectity file::

        A A 0 1
        B B 0 1
        C C 0 1

    Parameters
    ----------
    configs : dict
        The configuration dictionary.
        Must contain keys "NetworkFile" and "NetworkType".
    patch_df : DataFrame
        A dataframe containing populations of patches.

    Returns
    -------
    ndarray shape=(NumThetaIndices x NumPatches x NumPatches)
        The dynamic patch connectivity network
    """
    theta_df = pd.read_csv(
        configs["NetworkFile"],
        names=["src_Id", "dest_Id", "theta_index", "flow"],
        delimiter=" ",
        dtype={"src_Id": str, "dest_Id": str},
    )

    if configs["NetworkType"] == "Static":
        if not np.all(theta_df.theta_index == 0):
            raise ValueError("Theta indices mismatch. Ensure NetworkType=Static.")
    elif configs["NetworkType"] == "Weekly":
        if not list(sorted(set(theta_df.theta_index))) == list(range(53)):
            raise ValueError("Theta indices mismatch. Ensure NetworkType=Weekly.")
    elif configs["NetworkType"] == "Monthly":
        if not list(sorted(set(theta_df.theta_index))) == list(range(12)):
            raise ValueError("Theta indices mismatch. Ensure NetworkType=Monthly.")
    else:
        raise ValueError("Unknown NetworkType=%s" % configs["NetworkType"])

    Theta_indices = theta_df.theta_index.unique()
    Theta = np.zeros((len(Theta_indices), len(patch_df), len(patch_df)))
    patch_idx = {id_: i for i, id_ in enumerate(patch_df["id"])}
    for src_Id, dest_Id, theta_index, flow in theta_df.itertuples(
        index=False, name=None
    ):
        try:
            src_Idx = patch_idx[src_Id]
            dest_Idx = patch_idx[dest_Id]
            Theta[theta_index, src_Idx, dest_Idx] = flow
        except KeyError:
            logger.warning(
                "Ignoring flow entries for missing patches. Ensure all patches listed in PatchFile."
            )

    #print(f'Theta {type(Theta)} shape {Theta.shape}')
    logger.info("Loaded temporal travel matrix")
    return Theta


def do_patchsim_stoch_mobility_step(
    State_Array, patch_df, params, theta, seeds, vaxs, t
):
    """Do step of the stochastic (mobility) simulation."""
    S, E, A, I, U,H, R, D, V, peaks,new_inf = State_Array  ## Aliases for the State Array

    # seeding for day t (seeding implies S->I)
    actual_seed = np.minimum(seeds[t], S[t])
    S[t] = S[t] - actual_seed
    #I[t] = I[t] + params["symprob"]*actual_seed
    #A[t] = A[t] + (1-params["symprob"])*actual_seed
    I[t] = I[t] + actual_seed
    A[t] = A[t]# + (1-params["symprob"])*actual_seed
    # vaccination for day t
    actual_vax = np.minimum(vaxs[t] * params["vaxeff"], S[t])
    S[t] = S[t] - actual_vax
    V[t] = V[t] + actual_vax

    N = patch_df.pops.to_numpy()
    #print(N)
    #np.savetxt("fooN.csv", N, delimiter=",")
    # Effective population after movement step
    N_eff = theta.T.dot(N)
    for i in range (len(N_eff)):
        if N_eff[i]<1:
            N_eff[i]=1
    #np.savetxt("foo.csv", N_eff, delimiter=",")
    #N_eff.to_csv('Neff.csv')
    I_eff = theta.T.dot(I[t]+(1 - params["kappa"])*A[t])
    E_eff = theta.T.dot(E[t])

    # Force of infection from symp/asymptomatic individuals
    beta_j_eff = I_eff
    beta_j_eff = beta_j_eff / N_eff
    beta_j_eff = beta_j_eff *  params["beta2"] * params["beta"][:, t]
    beta_j_eff = beta_j_eff #* (
       # (1 - params["kappa"]) * (1 - params["symprob"]) + params["symprob"]
    #)
    beta_j_eff = np.nan_to_num(beta_j_eff)

    # Force of infection from presymptomatic individuals
    E_beta_j_eff = E_eff
    E_beta_j_eff = E_beta_j_eff / N_eff
    E_beta_j_eff = E_beta_j_eff * params["beta"][:, t]
    E_beta_j_eff = E_beta_j_eff * (1 - params["epsilon"])
    E_beta_j_eff = np.nan_to_num(E_beta_j_eff)

    
   
    # Infection force
    inf_force = theta.dot(beta_j_eff + E_beta_j_eff)

    # New exposures during day t
    #print(len(inf_force),len(S[0]))
    new_inf[t] = inf_force * S[t]
    #print(new_inf[t])
    new_inf[t] = np.minimum(new_inf[t], S[t])

    # Update to include presymptomatic and asymptomatic terms

    S[t + 1] = S[t] - new_inf[t]# + params["delta"] * R[t]#-D[t]
    
    E[t + 1] = new_inf[t] + (1 - params["alpha"]) * E[t]
    
    A[t + 1] = params["alpha"]*(1 - params["symprob"])*E[t]+(1-params["gamma_ar"]*params["sigma_ar"])*A[t]
    
    I[t + 1] = params["alpha"] * params["symprob"] * E[t] + params["factor_I"] * I[t]
    
    R[t + 1] = params["gamma_ir"] *params["sigma_ir"]* I[t] + (1 - params["delta"]) * R[t] + params["gamma_ar"] *params["sigma_ar"] * A[t] + params["sigma_hr"]*params["gamma_hr"]*H[t]

    H[t + 1] = params["sigma_ih"]*params["gamma_ih"]*I[t]+ params["sigma_uh"]*params["gamma_uh"]*U[t] + params["factor_H"]*H[t]
    
    U[t + 1] = params["sigma_iu"]*params["gamma_iu"]*I[t]+ params["sigma_hu"]*params["gamma_hu"]*H[t]+ params["factor_U"]*U[t]
    
    D[t + 1] = params["gamma_id"] *(1 - params["sigma_ir"]-params["sigma_ih"]-params["sigma_iu"]) * I[t] +  params["gamma_ud"]*(1-params["sigma_uh"])*U[t]+params["gamma_hd"]*(1-params["sigma_hr"]-params["sigma_hu"])*H[t]+ D[t]

    V[t + 1] = V[t]

    peaks[t + 1]= beta_j_eff + E_beta_j_eff

    ## Earlier computation of force of infection included network sampling.
    ## Now only implementing only disease progression stochasticity

#     N = patch_df.pops.values
#     S_edge = np.concatenate(
#         [
#             np.random.multinomial(
#                 S[t][x], theta[x] / (theta[x].sum() + 10 ** -12)
#             ).reshape(1, len(N))
#             for x in range(len(N))
#         ],
#         axis=0,
#     )
#     E_edge = np.concatenate(
#         [
#             np.random.multinomial(
#                 E[t][x], theta[x] / (theta[x].sum() + 10 ** -12)
#             ).reshape(1, len(N))
#             for x in range(len(N))
#         ],
#         axis=0,
#     )
#     I_edge = np.concatenate(
#         [
#             np.random.multinomial(
#                 I[t][x], theta[x] / (theta[x].sum() + 10 ** -12)
#             ).reshape(1, len(N))
#             for x in range(len(N))
#         ],
#         axis=0,
#     )
#     R_edge = np.concatenate(
#         [
#             np.random.multinomial(
#                 R[t][x], theta[x] / (theta[x].sum() + 10 ** -12)
#             ).reshape(1, len(N))
#             for x in range(len(N))
#         ],
#         axis=0,
#     )
#     V_edge = np.concatenate(
#         [
#             np.random.multinomial(
#                 V[t][x], theta[x] / (theta[x].sum() + 10 ** -12)
#             ).reshape(1, len(N))
#             for x in range(len(N))
#         ],
#         axis=0,
#     )
#     N_edge = S_edge + E_edge + I_edge + R_edge + V_edge

#     N_eff = N_edge.sum(axis=0)
#     I_eff = I_edge.sum(axis=0)
#     beta_j_eff = np.nan_to_num(params["beta"][:, t] * (I_eff / N_eff))

#     actual_SE = np.concatenate(
#         [
#             np.random.binomial(S_edge[:, x], beta_j_eff[x]).reshape(len(N), 1)
#             for x in range(len(N))
#         ],
#         axis=1,
#     ).sum(axis=1)
#     actual_EI = np.random.binomial(E[t], params["alpha"])
#     actual_IR = np.random.binomial(I[t], params["gamma"])
#     actual_RS = np.random.binomial(R[t], params["delta"])

#     ### Update to include presymptomatic and asymptomatic terms
#     S[t + 1] = S[t] - actual_SE + actual_RS
#     E[t + 1] = E[t] + actual_SE - actual_EI
#     I[t + 1] = I[t] + actual_EI - actual_IR
#     R[t + 1] = R[t] + actual_IR - actual_RS
#     V[t + 1] = V[t]


def do_patchsim_det_mobility_step(State_Array, patch_df, params, theta, seeds, vaxs, t):
    """Do step of the deterministic simulation."""
    S, E, A, I, U,H, R, D, V, peaks,new_inf = State_Array  ## Aliases for the State Array
    # nn = np.load('state_array_bloque1.npy')
    # nn0, nn9 = nn[0], nn[10]
    # print(f'{nn.shape} {new_inf[t].shape}')
    # print(f'en t={t} Infectados iniciales iguales: {(nn9 == new_inf[0]).sum()}')
    # print(new_inf[0][:6])
    # print(f'en t={t} Susceptibles iniciales iguales: {(nn0 ==   S[0]).sum()}')
    # print(f'en t={t} {(nn0 == S[t]).sum()}')
    # seeding for day t (seeding implies S->I)
    actual_seed = np.minimum(seeds[t], S[t])
    S[t] = S[t] - actual_seed
    #I[t] = I[t] + params["symprob"]*actual_seed
    #A[t] = A[t] + (1-params["symprob"])*actual_seed
    I[t] = I[t] + actual_seed
    A[t] = A[t]# + (1-params["symprob"])*actual_seed
    # vaccination for day t
    actual_vax = np.minimum(vaxs[t] * params["vaxeff"], S[t])
    S[t] = S[t] - actual_vax
    V[t] = V[t] + actual_vax

    N = patch_df.pops.to_numpy()
    #print(N)
    #np.savetxt("fooN.csv", N, delimiter=",")
    # Effective population after movement step
    N_eff = theta.T.dot(N)
    for i in range (len(N_eff)):
        if N_eff[i]<1:
            N_eff[i]=1
    #np.savetxt("foo.csv", N_eff, delimiter=",")
    #N_eff.to_csv('Neff.csv')
    I_eff = theta.T.dot(I[t]+(1 - params["kappa"])*A[t]) 
    # print(f'Infectados efectivos {I_eff}')    
    E_eff = theta.T.dot(E[t])
    # print(f'Expuestos efectivos {E_eff}')

    # Force of infection from symp/asymptomatic individuals
    beta_j_eff = I_eff
    beta_j_eff = beta_j_eff / N_eff
    
    # R_0 = params["beta2"] * params["beta"][:, t] * (1 / params["alpha"])
    # R_0 = params["beta2"] * params["beta"][:, t] * params["alpha"]
    # print(f'DAY {t}. type {type(R_0)} {R_0.shape} \n {np.mean(R_0, axis=0)}')
    # print(f'DAY {t}. {R_0.shape} --- {np.mean(R_0, axis=0)}')

    beta_j_eff = beta_j_eff *  params["beta2"] * params["beta"][:, t]
    #print(f'{t} beta_j_eff\n{beta_j_eff}')
    # beta_j_eff = beta_j_eff 
    ###beta_j_eff = beta_j_eff * ((1 - params["kappa"]) * (1 - params["symprob"]) + params["symprob"])

    beta_j_eff = np.nan_to_num(beta_j_eff)

    # Force of infection from presymptomatic individuals (POR DEFECTO =0)
    E_beta_j_eff = E_eff
    E_beta_j_eff = E_beta_j_eff / N_eff
    E_beta_j_eff = E_beta_j_eff * params["beta"][:, t]
    E_beta_j_eff = E_beta_j_eff * (1 - params["epsilon"])
    E_beta_j_eff = np.nan_to_num(E_beta_j_eff)
    #print(f'type {type(E_beta_j_eff)} {E_beta_j_eff.shape} \n {E_beta_j_eff}')

    
    
    # Infection force
    inf_force = theta.dot(beta_j_eff + E_beta_j_eff) 
    ###inf_force = beta_j_eff

    # New exposures during day t
    #print(len(inf_force),len(S[0]))
    # print(inf_force)
    # print(len(inf_force))
    # print(len(S[t]))
    # plt.plot(inf_force)
    # plt.show()
    # new_inf_pre[t] = new_inf[t]
    # if t>3:
    # print(f'Antes de mult. inf. force {new_inf[0][:6]}')
    # print(f'inf. force {inf_force[:6]}')
    # print(f' Susceptibles iguales : {(nn0 ==   S[0]).sum()}')
    new_inf[t] = inf_force * S[t]
    # print(f'Despp de mult. inf. force {new_inf[0][:6]}')
    # new_inf[t] = np.minimum(new_inf_pre[t], new_inf[t]) 
    # print(f'en t={t} Infectados iniciales iguales luego: {(nn9 == new_inf[0]).sum()}')
    
    # if t>=300 :
    #     print(f'{t} new infected\n{new_inf[310:320,0]}')
    #     print(f'suscep\n {S[310:320,0]}')
    
    new_inf[t] = np.minimum(new_inf[t], S[t]) 

    # Update to include presymptomatic and asymptomatic terms

    #print( params["delta"])
    S[t + 1] = S[t] - new_inf[t] + params["delta"] * R[t]#-D[t]
    
    E[t + 1] = new_inf[t] + (1 - params["alpha"]) * E[t]
    
    A[t + 1] = params["alpha"]*(1 - params["symprob"])*E[t]+(1-params["gamma_ar"]*params["sigma_ar"])*A[t]
    
    I[t + 1] = params["alpha"] * params["symprob"] * E[t] + params["factor_I"] * I[t]
    
    R[t + 1] = params["gamma_ir"] *params["sigma_ir"]* I[t] + (1 - params["delta"]) * R[t] + params["gamma_ar"] *params["sigma_ar"] * A[t] + params["sigma_hr"]*params["gamma_hr"]*H[t]

    H[t + 1] = params["sigma_ih"]*params["gamma_ih"]*I[t]+ params["sigma_uh"]*params["gamma_uh"]*U[t] + params["factor_H"]*H[t]
    
    U[t + 1] = params["sigma_iu"]*params["gamma_iu"]*I[t]+ params["sigma_hu"]*params["gamma_hu"]*H[t]+ params["factor_U"]*U[t]
    
    D[t + 1] = params["gamma_id"] *(1 - params["sigma_ir"]-params["sigma_ih"]-params["sigma_iu"]) * I[t] +  params["gamma_ud"]*(1-params["sigma_uh"])*U[t]+params["gamma_hd"]*(1-params["sigma_hr"]-params["sigma_hu"])*H[t]+ D[t]

    V[t + 1] = V[t]

    #peaks[t + 1] = beta_j_eff + E_beta_j_eff
    # peaks[t + 1] = D[t+1] #original
    peaks[t ] = inf_force
    # peaks[t + 1] = params["beta2"] * params["beta"][:, t] * params["alpha"]

    

    #return (beta_j_eff + E_beta_j_eff)

    

def do_patchsim_det_force_step(State_Array, patch_df, params, theta, seeds, vaxs, t):
    """Do step of the deterministic simulation."""
    S, E, A ,I, R, V, new_inf = State_Array  ## Aliases for the State Array

    # seeding for day t (seeding implies S->I)
    actual_seed = np.minimum(seeds[t], S[t])
    S[t] = S[t] - actual_seed
    I[t] = I[t] + actual_seed

    # vaccination for day t
    actual_vax = np.minimum(vaxs[t] * params["vaxeff"], S[t])
    S[t] = S[t] - actual_vax
    V[t] = V[t] + actual_vax

    N = patch_df.pops.to_numpy()

    # Effective beta
    beta_j_eff = I[t]
    beta_j_eff = beta_j_eff / N
    beta_j_eff = beta_j_eff * params["beta"][:, t]
    beta_j_eff = np.nan_to_num(beta_j_eff)

    # Infection force
    inf_force = theta.T.dot(beta_j_eff)

    # New exposures during day t
    new_inf[t] = inf_force * S[t]
    new_inf[t] = np.minimum(new_inf[t], S[t])

    # Update to include presymptomatic and asymptomatic terms
    S[t + 1] = S[t] - new_inf[t] + params["delta"] * R[t]
    E[t + 1] = new_inf[t] + (1 - params["alpha"]) * E[t]
    I[t + 1] = params["alpha"] * E[t] + (1 - params["gamma"]) * I[t]
    R[t + 1] = params["gamma"] * I[t] + (1 - params["delta"]) * R[t]
    V[t + 1] = V[t]


def patchsim_step(State_Array, patch_df, configs, params, theta, seeds, vaxs, t, stoch):
    """Do step of the simulation."""
    if stoch:
        if configs["Model"] == "Mobility":
            return do_patchsim_stoch_mobility_step(
                State_Array, patch_df, params, theta, seeds, vaxs, t
            )
        else:
            raise ValueError(
                "Unknown Model %s for stochastic simulation" % configs["Model"]
            )
    else:
        if configs["Model"] == "Mobility":
            return do_patchsim_det_mobility_step(
                State_Array, patch_df, params, theta, seeds, vaxs, t
            )
        elif configs["Model"] == "Force":
            return do_patchsim_det_force_step(
                State_Array, patch_df, params, theta, seeds, vaxs, t
            )
        else:
            raise ValueError(
                "Unknown Model %s for deterministic simulation" % configs["Model"]
            )


def epicurves_todf(configs, params, patch_df, State_Array,compartment):
    """Convert the epicurve (new infection over time) into a dataframe.

    Parameters
    ----------
    configs : dict
        The configuration dictionary.
    params : dict
        A dictionary of model parameters.
    patch_df : dict
        A dataframe containing populations of patches.
    State_Array : 5 tuple
        A tuple of disease state information.

    Returns
    -------
    DataFrame
        A dataframe containing the new infections.
        There is one row per patch.
        There is one column per timestep.
    """
    #new_inf = State_Array[-1]
    new_inf = State_Array[compartment]
    # print(f'in outputfile: {new_inf[0][:6]}')

    data = new_inf[:-1, :].T
    data = data * float(params["scaling"])
    if configs["OutputFormat"] == "Whole":
        data = data.round().astype(int)

    index = patch_df.id
    columns = np.arange(int(configs["Duration"]))

    out_df = pd.DataFrame(index=index, columns=columns, data=data)
    return out_df


def write_epicurves(configs, params, patch_df, State_Array, write_epi, return_epi,compartment):
    """Write the epicurve into the output file.

    Parameters
    ----------
    configs : dict
        The configuration dictionary.
    params : dict
        A dictionary of model parameters.
    patch_df : dict
        A dataframe containing populations of patches.
    State_Array : 5 tuple
        A tuple of disease state information.
    write_epi : bool
        If true write the epicurve to configs[OutputFile]
    return_epi : bool
        If true return the whole epicurve dataframe.
        Otherwise return the total number of people infected.

    Returns
    -------
    number or DataFrame
        If return_epi is true return the whole epicurve dataframe.
        Otherwise return the total number of people infected.
    """
    out_df = epicurves_todf(configs, params, patch_df, State_Array,compartment)

    if write_epi:
        out_df.to_csv(configs["OutputFile"], header=None, sep=" ")

    if return_epi:
        return out_df
    else:
        return out_df.sum().sum()


def dummy_intervene_step(configs, patch_df, params, Theta, seeds, vaxs, t, State_Array=None):
    """Run a dummy intervention step.

    configs : dict
        The configuration dictionary.
    patch_df : dict
        A dataframe containing populations of patches.
    params : dict
        A dictionary of model parameters.
    Theta : ndarray shape=(NumThetaIndices x NumPatches x NumPatches)
        The dynamic patch connectivity network
    seeds : ndarray shape=(NumTimsteps x NumPatches)
        A seeding schedule matrix
    vaxs : ndarray shape=(NumTimsteps x NumPatches)
        A vaccination schedule matrix (NumTimsteps x NumPatches)
    t : int
        The timestep that was just finished.
    """


def run_disease_simulation(
    configs,
    patch_df=None,
    params=None,
    Theta=None,
    seeds=None,
    vaxs=None,
    return_epi=False,
    write_epi=False,
    log_to_file=True,
    intervene_step=None,compartment=None
):
    """Run the disease simulation.

    Parameters
    ----------
    configs : dict
        The configuration dictionary.
    patch_df : dict, optional
        A dataframe containing populations of patches.
    params : dict, optional
        A dictionary of model parameters.
    Theta : ndarray shape=(NumThetaIndices x NumPatches x NumPatches), optional
        The dynamic patch connectivity network
    seeds : ndarray shape=(NumTimsteps x NumPatches), optional
        A seeding schedule matrix
    vaxs : ndarray shape=(NumTimsteps x NumPatches), optional
        A vaccination schedule matrix (NumTimsteps x NumPatches)
    write_epi : bool
        If true write the epicurve to configs[OutputFile]
    return_epi : bool
        If true return the whole epicurve dataframe.
        Otherwise return the total number of people infected.
    log_to_file : bool
        If true register a new logging handler to configs[LogFile]
        Also removes any other file handler previously registered.
    intervene_step : function, optional
        If intervene_step step is not None,
        it is called after every step.
        It is expected to have the same signature as dummy_intervene_step.

    Returns
    -------
    DataFrame
        A dataframe containing the new infections.
        There is one row per patch.
        There is one column per timestep.
    """
    if log_to_file:
        handler = logging.FileHandler(configs["LogFile"], mode="a")
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        # remove the existing file handlers
        for hdlr in logger.handlers[:]:
            if isinstance(hdlr, logging.FileHandler):
                logger.removeHandler(hdlr)

        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    logger.info("Starting PatchSim")
    start = time.time()

    logger.info("Operating PatchSim under %s Model", configs["Model"])

    if patch_df is None:
        patch_df = load_patch(configs)
    if params is None:
        params = load_params(configs, patch_df)
    if Theta is None:
        Theta = load_Theta(configs, patch_df)
    if seeds is None:
        seeds = load_seed(configs, params, patch_df)
    if vaxs is None:
        vaxs = load_vax(configs, params, patch_df)

    logger.info("Initializing simulation run...")

    if "RandomSeed" in configs:
        np.random.seed(int(configs["RandomSeed"]))
        stoch = True
        logger.info("Found RandomSeed. Running in stochastic mode...")
    else:
        stoch = False
        logger.info("No RandomSeed found. Running in deterministic mode...")

    # Number of states (SEAIRHUV) + One for tracking new infections
    dim = 10 + 1
    shape = (dim, params["T"] + 1, len(patch_df))
    # shape = (dim, params["T"] , len(patch_df))
    if stoch:
        State_Array = np.zeros(shape, dtype=int)
    else:
        State_Array = np.zeros(shape, dtype=float)

    if configs["LoadState"] == "True":
        # Load all
        State_Array[:, 0, :] = np.load(configs["LoadFile"])
        #print(f'{State_Array[10,0,:].shape}')
    else:
        # Load only the Succeptiables
        State_Array[0, :, :] = patch_df.pops.to_numpy()

    if configs["NetworkType"] == "Static":
        for t in range(params["T"]):
            patchsim_step(
                State_Array, patch_df, configs, params, Theta[0], seeds, vaxs, t, stoch
            )

            if intervene_step is not None:
                intervene_step(configs, patch_df, params, Theta, seeds, vaxs, t, State_Array)

    elif configs["NetworkType"] == "Weekly":
        ref_date = datetime.strptime("Jan 6 2020", "%b %d %Y")  # 
        # ref_date = datetime.strptime("Dec 30 2019", "%b %d %Y")  # 
        for t in range(params["T"]):
            curr_date = ref_date + timedelta(days=t + int(configs["StartDate"]))            
            curr_week = int(curr_date.strftime("%U"))
            # print(curr_date, curr_week)
            # print(f'En t= {t}, curr_week = {curr_week}, current_week-1= {curr_week-1}')
            patchsim_step(
                State_Array,
                patch_df,
                configs,
                params,
                Theta[curr_week-1], 
                seeds,
                vaxs,
                t,
                stoch,
            )

            if intervene_step is not None:
                intervene_step(configs, patch_df, params, Theta, seeds, vaxs, t, State_Array)

    elif configs["NetworkType"] == "Monthly":
        ref_date = datetime.strptime("Jan 1 2020", "%b %d %Y")  # is a Sunday
        for t in range(params["T"]):
            curr_date = ref_date + timedelta(days=t + int(configs["StartDate"]))
            curr_month = int(curr_date.strftime("%m"))

            patchsim_step(
                State_Array,
                patch_df,
                configs,
                params,
                Theta[curr_month - 1],
                seeds,
                vaxs,
                t,
                stoch,
            )

            if intervene_step is not None:
                intervene_step(configs, patch_df, params, Theta, seeds, vaxs, t, State_Array)
    else:
        raise ValueError("Unknown NetworkType=%s" % configs["NetworkType"])

    if configs["SaveState"] == "True":
        logger.info("Saving StateArray to File")
        np.save(configs["SaveFile"], State_Array[:, :, :])

    elapsed = time.time() - start
    logger.info("Simulation complete. Time elapsed: %s seconds.", elapsed)

    return write_epicurves(
        configs, params, patch_df, State_Array, write_epi, return_epi,compartment
    )
