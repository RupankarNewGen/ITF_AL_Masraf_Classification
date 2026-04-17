def check_and_modify_labels(json_data, bottom_value,label):
    if label in json_data and bottom_value in json_data:
        json_data[bottom_value][0][0] = "~ "+json_data[bottom_value][0][0]
        
        #joined_value = "~ ".join([json_data[label][0][0], json_data["insurance_issuer_address_bottom"][0][0]])
        joined_value = json_data[label]+json_data[bottom_value]
        json_data[label] = joined_value
        del json_data[bottom_value]
    else:
        if bottom_value in json_data:
            json_data[label] = json_data.pop(bottom_value)

    return json_data


def  merge_top_bottom_keys(key_name_changes, my_dict):
    for old_key, new_key in key_name_changes.items():
        if old_key in my_dict:
            # Get the value associated with the old key
            value = my_dict[old_key]

            # Delete the old key-value pair
            del my_dict[old_key]

            # Add the new key-value pair
            my_dict[new_key] = value
    return my_dict


