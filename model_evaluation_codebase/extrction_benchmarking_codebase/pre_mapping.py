PredictionKeyMapping: dict = {
    "PL":
    {"consignee_name": "remitter_importer_buyer_name",
        "consignee_address": "remitter_importer_buyer_address",
        "shipper_name": "consignor_name",
        "shipper_address": "consignor_address"
        },
    "CI":
        {
    "consignee_name": "remitter_importer_buyer_name",
    "consignee_address": "remitter_importer_buyer_address",
    "shipper_name": "consignor_name", 
    "shipper_address": "consignor_address",
    }
}


ParentKeyMapping: dict = {
    'CS':{
        "currency_amount":["csh_bill_currency", "csh_bill_amount"]
    }
}

OverLappingKeys:dict = {
    'IC':{
        "sum_insured_amount": "sum_insured_currency"
    }
}

top_value: list = ["drawer_bank_address",'drawer_bank_name','drawer_bank_bic', 'insurance_issuer_address', 'insurance_issuer_name']
bottom_value: list = ["drawer_bank_bottom_address",'drawer_bank_bottom_name','drawer_bank_bottom_bic', 'insurance_issuer_address_bottom', 'insurance_issuer_name_bottom']

weights_fields : list = ['net_weight', 'gross_weight','total_quantity_of_goods', \
    'tolerance_of_quantity', 'quantity', 'unit', 'rate', 'net_weight', 'unit_price']

amount_fields : list = ['total_amount_in_numeric', 'tax_amount', 'amount_due', \
                        'advance_amount', 'total_amount_in_figure', \
                            'amount_insurance','boe_amount', 'invoice_amount', 'transaction_amonunt_value',\
                                'invoice_discount_amount', 'invoice_tax_amount',\
                                    'premium_amount', 'total_amount_in_numeric', \
                                        'amount_due', 'advance_amount', 'total_amount_in_figure', 'tax_amount']

currency_fields : list = ['currency_in_numeric', 'boe_currency', \
                'invoice_currency', 'transaction_currency', 'premium_currency', 'currency_in_numeric', 'currency']

transport_fields : list = ['pre_carriage_by', 'mode_of_transport', 'means_of_transport', \
                    'pre_carriage_by', 'mode_of_dispatch']