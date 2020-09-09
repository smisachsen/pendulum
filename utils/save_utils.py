import os

def create_folder_if_not_exists(path):
    if os.path.exists(path):
        return
    os.mkdir(path)


def get_new_folder(path):
    create_folder_if_not_exists(path)

    current_folders = os.listdir(path)
    new_folder_index = str(len(current_folders))

    new_folder_path = os.path.join(path, new_folder_index)
    os.mkdir(new_folder_path)

    return new_folder_path


if __name__ == '__main__':
    foldername = "testfolder/"
    new_folder_path = get_new_folder(foldername)

    print(new_folder_path)
