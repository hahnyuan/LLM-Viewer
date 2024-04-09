import axios from 'axios'

function update_avaliable_model_hardwares(avaliable_hardwares, avaliable_model_ids, ip_port) {
    const url = 'http://' + ip_port.value + '/get_avaliable'
    axios.get(url).then(function (response) {
        console.log("update_avaliable_model_hardwares",response);
        avaliable_hardwares.value = response.data.avaliable_hardwares
        avaliable_model_ids.value = response.data.avaliable_model_ids
    })
        .catch(function (error) {
            console.log("error in get_avaliable");
            console.log(error);
        });
}

async function update_frontend_params_info(frontend_params_info, model_id, ip_port) {
    const url = 'http://' + ip_port.value + '/get_frontend_params_info'
    
    try {
        const response = await axios.post(url, {
            model_id: model_id.value
        });
        console.log("update_frontend_params_info",response);
        
        frontend_params_info.value = response.data.frontend_params_info
        for (let param_info of frontend_params_info.value) {
            console.log(param_info);
            param_info.value = param_info.default
        }
    } catch (error) {
        console.log("error in get_frontend_params_info");
        console.log(error);
    }

}

export { update_avaliable_model_hardwares, update_frontend_params_info }