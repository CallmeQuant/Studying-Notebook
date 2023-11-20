import torch
from ..modules.models import GRU, TCN, Transformer

# Setting up configurations
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")

def setup_model_config(config_dict):
    config_gru = config_dict.copy()
    config_gru["USE_ATTENTION"] = False
    model_gru = GRU(config_gru["N_LAYER"],
                config_gru["HIDDEN_DIM"],
                config_gru["SEQ_LEN"],
                config_gru["N_FEAT"],
                config_gru["DROPOUT"],
                config_gru["BIDIRECTIONAL"],
                config_gru["USE_ATTENTION"],
                config_gru['LB'], config_gru['UB']
            ).to(device)

    model_gru_aa = GRU(config_dict["N_LAYER"],
                       config_dict["HIDDEN_DIM"],
                       config_dict["SEQ_LEN"],
                       config_dict["N_FEAT"],
                       config_dict["DROPOUT"],
                       config_dict["BIDIRECTIONAL"],
                       config_dict["USE_ATTENTION"],
                       config_dict['LB'], config_dict['UB']
                       ).to(device)

    hidden_size = 5
    level = 3
    num_channels = [hidden_size] * (level - 1) + [config_dict['TCN']['SEQ_LEN']]

    model_tcn = TCN(config_dict['TCN']['N_FEAT'],
                    config_dict['TCN']['N_OUT'],
                    num_channels,
                    config_dict['TCN']["KERNEL_SIZE"],
                    config_dict['TCN']["DROPOUT"],
                    config_dict['TCN']["SEQ_LEN"],
                    config_dict['LB'], config_dict['UB'],
                    ).to(device)

    model_transformer = Transformer(
                config_dict['TRANSFORMER']['N_FEAT'],
                config_dict['TRANSFORMER']['SEQ_LEN'],
                config_dict['TRANSFORMER']["N_LAYER"],
                config_dict['TRANSFORMER']["N_HEAD"],
                config_dict['TRANSFORMER']["DROPOUT"],
                config_dict['TRANSFORMER']["N_OUT"],
                config_dict['LB'], config_dict['UB']
            ).to(device)

    model_names = ['gru', 'gru_aa', 'tcn', 'transformer']
    model_list = [model_gru, model_gru_aa, model_tcn, model_transformer]
    model_dict =  {name: model for name, model in zip(model_names, model_list)}

    return model_names, model_list, model_dict

