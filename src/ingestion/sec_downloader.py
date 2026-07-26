import json
import requests
import time
import os
header = {"User-Agent": "Prabhav prabhavpradeep046@gmail.com"}
def get_CIK_ticker(ticker:str) -> str:
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=header)
    if response.status_code!=200:
        raise Exception(f"Failed to fetch tickers: {response.status_code}")
    data = response.json()
    for item in data.values():
        if item["ticker"].upper() == ticker.upper():
            return str(item["cik_str"]).zfill(10)
    raise ValueError(f"Ticker '{ticker}' not found.")
def get_company_info(cik:str)->dict:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=header)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}")
def get_tags(data:dict) -> list:
    results = []
    for i in range(len(data["filings"]["recent"]["form"])):
        if(data["filings"]["recent"]["form"][i] in ["10-K","10-Q"]):
            accessionNumber = data["filings"]["recent"]["accessionNumber"][i]
            primaryDocument = data["filings"]["recent"]["primaryDocument"][i]
            info = {
                "accessionNumber":accessionNumber,
                "primaryDocument":primaryDocument
            }
            results.append(info)
    return results
def download_htms(results:list,cik:str,save_dir: str = "data/raw_filings") -> list:
    os.makedirs(save_dir, exist_ok=True)
    saved_file_paths = []
    for item in results:
        
        accession = item["accessionNumber"]
        primary = item["primaryDocument"]
        accession = accession.replace("-","")
        filename = f"{cik}_{accession}_{primary}"
        filepath = os.path.join(save_dir, filename)
        
        if os.path.exists(filepath):
            print(f"Skipping (already downloaded): {filepath}")
            saved_file_paths.append(filepath)
            continue
        time.sleep(0.12)
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary}"
        response = requests.get(url,headers = header)
        if response.status_code == 200:
            filename = f"{cik}_{accession}_{primary}"
            filepath = os.path.join(save_dir, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(response.text)
                
            saved_file_paths.append(filepath)
            print(f"Saved: {filepath}")
        else:
            raise Exception(f"Failed to fetch {url}: {response.status_code}")
            
    return saved_file_paths

if __name__ == "__main__":
    target_ticker = "AAPL"
    print(f"Starting pipeline for {target_ticker}...")
    
    # Step 1: Get CIK
    cik = get_CIK_ticker(target_ticker)
    print(f"Found CIK: {cik}")
    
    # Step 2: Get Submissions Metadata
    metadata = get_company_info(cik)
    
    # Step 3: Extract 10-K and 10-Q links
    # For the pilot test, let's only grab the 3 most recent filings so it runs fast
    filing_links = get_tags(metadata)[:3] 
    print(f"Found {len(filing_links)} filings to download.")
    
    # Step 4: Download and Save
    saved_files = download_htms(filing_links, cik)
    print("Pilot download complete!")