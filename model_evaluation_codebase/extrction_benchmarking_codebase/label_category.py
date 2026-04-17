import json
from configparser import ConfigParser
product_wise_folder = ConfigParser()
product_wise_folder.read("/home/ntlpt19/Downloads/Evaluation_Data/updated_code/post_processing/post_processing/config.ini")


class LabelCategory:
    def __init__(self):
        # self.category_name = category_name
        self.single_line_fields = []
        self.multi_line_fields = []
        self.single_word_fields = []
        self.date_fields = []
        self.critical_fields = []
        self.numeric_fields = []
        self.master_fields = []
        self.merge_fields = []
        self.address_fields = []
        self.remove_fields = []
    def add_single_line_fields(self, field_name):
        self.single_line_fields.append(field_name)
        
    def add_address_fields(self, field_name):
        self.address_fields.append(field_name)
        
    def add_multi_line_fields(self, field_name):
        self.multi_line_fields.append(field_name)

    def add_single_word_fields(self, field_name):
        self.single_word_fields.append(field_name)

    def add_date_fields(self, field_name):
        self.date_fields.append(field_name)

    def add_critical_fields(self, field_name):
        self.critical_fields.append(field_name)

    def add_numeric_fields(self, field_name):
        self.numeric_fields.append(field_name)

    def add_master_fields(self, field_name):
        self.master_fields.append(field_name)

    def add_merge_fields(self, field_name):
        self.merge_fields.append(field_name)

    def add_remove_fields(self, field_name):
        self.remove_fields.append(field_name)



    def display_category_summary(self):
        # print(f"Category: {self.category_name}")
        print(f"Single Line Fields: {', '.join(self.single_line_fields)}")
        print(f"Multi Line Fields: {', '.join(self.multi_line_fields)}")
        print(f"Single Word Fields: {', '.join(self.single_word_fields)}")
        print(f"Date Fields: {', '.join(self.date_fields)}")
        print(f"Critical Fields: {', '.join(self.critical_fields)}")
        print(f"Numeric Fields: {', '.join(self.numeric_fields)}")
        print(f"Master Fields: {', '.join(self.master_fields)}")
        print(f"Merge Fields: {', '.join(self.merge_fields)}")
        print(f"address_fields: {', '.join(self.address_fields)}")
        print(f"remove_fields: {', '.join(self.remove_fields)}")

# # Example Usage:\

# label_category = LabelCategory("PackingList")
# with open(str(product_wise_folder['TransportDocument']['PackingList']), 'r') as file:
#     pl_keys = json.load(file)
# for method, fields in pl_keys.items():
#     for field in fields:
#         getattr(label_category, f'add_{method}')(field)


# label_category.display_category_summary()
