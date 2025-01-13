from glob import glob
import subprocess
import jsonlines
import logging
import uuid
import json
import os

def restart_container(container_name, apptainer_image = "docker://node:latest") -> bool:
    command = f"apptainer instance stop {container_name} && apptainer instance start {apptainer_image} {container_name}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        return True
    raise KeyError(f"[!] Fatal Error: restart failed for container {container_name}")

def create_apptainer_container(container_name ="eval_js_container",
                               apptainer_image = "docker://node:latest"):
    # check docker is installed
    check_process = subprocess.run(["apptainer", "--version"], capture_output=True, text=True)
    if check_process.returncode != 0:
        raise Exception("[!] Apptainer is not installed")
    # check container exists
    check_process = subprocess.run(["apptainer", "instance", "list"], capture_output=True, text=True)
    if check_process.returncode == 0 and container_name in check_process.stdout:
        # Container already exists, restart it
        logging.info(f"[+] Container '{container_name}' already exists, restarting it")
        restart_container(container_name)
        return container_name
    # Create an apptainer container and keep it running
    logging.info(f"[+] Creating a container '{container_name}' from image '{apptainer_image}'")
    subprocess.run(["apptainer", "instance", "start", apptainer_image, container_name], check=True)
    return container_name

def stop_apptainer_container(container_name) -> bool:
    result = subprocess.run(f"apptainer instance stop {container_name}", shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"[!] Error: Failed to stop the container '{container_name}'")
        return False
    return True

def compile_and_run_JS_code_in_docker(container_name: str, code: str, program_inputs: list, expected_outputs: list, timeout: int = 5):
    file_name = f"temp_{uuid.uuid4()}.js"

    # Cleanup: remove the existing file
    if os.path.exists(file_name):
        os.remove(file_name)

    try:
        # Unique ID for the file name to avoid conflicts
        with open(file_name, "w") as file:
            file.write(code)

    except subprocess.CalledProcessError as e:
        logging.error(f"[!] Error during operations:\n{e}")
        return 0

    # Run the compiled program
    results = []
    for program_input, expected_output in zip(program_inputs, expected_outputs):
        try:
            # print("HHIII1")
            run_process = subprocess.run(
                ["apptainer", "exec", f"instance://{container_name}", "node", f"{file_name}"],
                input=program_input,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            # print("HHIII2")
            # Check the output
            if run_process.returncode != 0:
                logging.debug(f"[!] Runtime Error: {run_process.stderr}")

            output = run_process.stdout
            if output.strip() == expected_output.strip():
                logging.debug("[+] Success: Output matches expected output") 
                results.append(1)
            else:
                logging.debug(f"[-] Failure: Output does not match. Expected '{expected_output}', got '{output}'") 
                results.append(0)
                break
        except subprocess.TimeoutExpired:
            logging.debug("[-] Error: Program timeout")
            results.append(0)
            break
    # Cleanup: remove the temporary file
    if os.path.exists(file_name):
        os.remove(file_name)
    return 0 if 0 in results else 1

if __name__ == "__main__":
    # Create the Docker container
    container_name = create_apptainer_container(container_name ="eval_js_container", docker_image ="docker://node:latest")
    # get program inputs and expected outputs
    test_cases = {}
    with jsonlines.open('./datasets/codenet_dataset/Project_CodeNet_selected.jsonl','r') as reader:
        for obj in reader:
            filename = obj['file_dir']
            test_cases[filename] = obj['test_case']
    # eval each JS code
    JS_code_path = "./datasets/codenet_dataset/"
    files = glob(os.path.join(JS_code_path, "*.js"))
    for fp in files:
        with open(fp, "r") as f:
            code = f.read()
        k = os.path.basename(fp).split("_")[-1].removesuffix(".js")
        test_case = test_cases[k]
        program_inputs = [i[0] for i in test_case]
        expected_outputs = [i[1] for i in test_case]
        res = compile_and_run_JS_code_in_docker(container_name, code, program_inputs, expected_outputs)
        print(fp,res)