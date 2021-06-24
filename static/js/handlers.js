function load_data_handler() {
    var formData = new FormData();

    var xhr = new XMLHttpRequest();
    xhr.open('GET', '/api/process', true);

    xhr.onload = function (e) {
        if (xhr.status === 200) {
            // console.log('uploaded');
            process_handler(JSON.parse(xhr.response));
        } else {
            alert('An error occurred!');
        }
    };
    xhr.send(formData);
}

function process_handler(data) {
    processed_data = data.data;
    statistic_component.redraw(processed_data['statistics'], processed_data['op_groups'], processed_data['schedule']);
    network_component.redraw(processed_data['network']);
    add_statistic_checkbox();
    add_intro();
}

