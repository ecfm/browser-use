# Goal: Search LinkedIn for companies and extract employee profiles for product manager, marketing, sales, and executive roles.

import asyncio
import os
import re
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, SecretStr

from browser_use import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig

# Load environment variables
load_dotenv()

# Validate required environment variables
api_key = os.getenv('DEEPSEEK_API_KEY')
if not api_key:
	raise ValueError('DEEPSEEK_API_KEY is not set in the environment variables. Please add it to your .env file.')


class CompanyInfo(BaseModel):
	name: str
	id: Optional[str] = None
	url: Optional[str] = None
	description: str = ''
	employee_count: str = ''
	profiles_extracted: int = 0
	relevant_profiles: int = 0
	search_status: str = 'pending'  # pending, completed, failed


class EmployeeInfo(BaseModel):
	name: str
	position: str
	company: str
	linkedin_url: str
	relevancy: bool = False


# Constants
COMPANY_SEARCH_LINK = 'https://www.linkedin.com/search/results/companies/?keywords='
PEOPLE_SEARCH_LINK = 'https://www.linkedin.com/search/results/people/?currentCompany=%5B%22<COMPANY_ID>%22%5D&origin=FACETED_SEARCH&schoolFilter=%5B%223147%22%2C%221503%22%2C%222650%22%5D&sid=eGt'
DATA_FILE_PATH = 'data/store of Appliances.txt'
EXCLUDE_FILES = ['data/store of Tools.txt', 'data/store of Electronics.txt']
DATA_NAME = 'appliances'
OUTPUT_FILE_PATH = f'results/{DATA_NAME}_linkedin_profiles.csv'
COMPANY_INFO_FILE_PATH = f'results/{DATA_NAME}_linkedin_companies.csv'


def extract_company_names() -> List[str]:
	"""Extract company names from the data file."""
	companies = []
	with open(DATA_FILE_PATH, 'r') as f:
		companies = set([line.strip() for line in f.readlines() if line.strip()])
	excluded_companies = set()
	for exclude_file in EXCLUDE_FILES:
		with open(exclude_file, 'r') as f:
			excluded_companies.update([line.strip() for line in f.readlines() if line.strip()])
	companies = [company for company in companies if company not in excluded_companies]
	return companies


async def find_company_id(browser_context: BrowserContext, company_name: str) -> Optional[CompanyInfo]:
	"""Search for a company and extract its LinkedIn ID."""
	company_info = CompanyInfo(name=company_name)

	# Search for the company
	search_url = f'{COMPANY_SEARCH_LINK}{company_name.replace(" ", "%20")}'
	page = await browser_context.get_current_page()
	await page.goto(search_url)
	await page.wait_for_load_state()
	print(f'Searching for company: {company_name}')

	# Check for "No results found" message
	no_results = await page.query_selector('h2.artdeco-empty-state__headline')
	if no_results:
		print(f'No results found for {company_name}')
		return None

	# Check if there are search results
	try:
		# Use a selector that matches the actual element structure
		company_selector = 'a[data-test-app-aware-link][href*="/company/"]'

		await page.wait_for_selector(company_selector, timeout=2000)  # Reduced timeout
		company_links = await page.query_selector_all(company_selector)

		if company_links and len(company_links) > 0:
			# Click the first company link
			await company_links[0].click()
			await page.wait_for_load_state()

			# Check if we're on a company page
			current_url = page.url
			company_info.url = current_url

			# Try to find the employee count link with company ID
			try:
				await page.wait_for_selector('a.org-top-card-summary-info-list__info-item-link', timeout=2000)  # Reduced timeout
				employee_links = await page.query_selector_all('a.org-top-card-summary-info-list__info-item-link')

				for link in employee_links:
					href = await link.get_attribute('href')
					if href and 'currentCompany' in href:
						# Extract company IDs from the URL - now handles multiple IDs
						match = re.search(r'currentCompany=%5B%22(\d+)%22(?:%2C%22(\d+)%22)*%5D', href)
						if match:
							# Get the first company ID (usually the main one)
							company_id = match.group(1)
							company_info.id = company_id
							# Get employee count
							try:
								employee_count_span = await link.query_selector('span.t-normal.t-black--light')
								if employee_count_span:
									company_info.employee_count = await employee_count_span.inner_text()
							except Exception as e:
								print(f'Could not get employee count: {str(e)}')
								company_info.employee_count = ''

							# Try to get company description
							try:
								description_element = await page.query_selector(
									'div.organization-about-module__content-consistant-cards-description'
								)
								if description_element:
									company_info.description = await description_element.inner_text()
							except Exception as e:
								print(f'Could not get company description: {str(e)}')
								company_info.description = ''

							return company_info
			except Exception as e:
				print(f'Could not find employee list for {company_name}: {str(e)}')
				return None
		else:
			print(f'No company links found for {company_name}')
			return None
	except Exception as e:
		print(f'No results found for {company_name}: {str(e)}')
		return None

	return None


async def collect_all_profiles(browser_context: BrowserContext, company_info: CompanyInfo) -> List[dict]:
	"""Collect all employee profiles from a company."""
	profiles = []
	processed_urls = set()  # Keep track of processed URLs to avoid duplicates
	MAX_PROFILES = 80  # Maximum number of profiles to collect per company

	if not company_info.id:
		return profiles

	# Navigate to the people search page
	people_search_url = PEOPLE_SEARCH_LINK.replace('<COMPANY_ID>', company_info.id)
	page = await browser_context.get_current_page()
	await page.goto(people_search_url)
	await page.wait_for_load_state()
	print(f'Searching for employees at {company_info.name}')

	# Wait for search results to load
	await asyncio.sleep(2)

	# Process the search results
	try:
		while True:
			current_url = page.url
			if current_url in processed_urls:
				print('Reached a previously processed page, stopping pagination')
				break
			processed_urls.add(current_url)

			# Check for "No results found" message first
			no_results = await page.query_selector('h2.artdeco-empty-state__headline')
			if no_results:
				no_results_text = await no_results.inner_text()
				if 'No results found' in no_results_text:
					print(f'No employee profiles found for {company_info.name}')
					return profiles

			# Find profile cards using data attributes
			try:
				await page.wait_for_selector('div[data-chameleon-result-urn*="urn:li:member"]', timeout=10000)
				profile_cards = await page.query_selector_all('div[data-chameleon-result-urn*="urn:li:member"]')
			except Exception as e:
				print(f'No profile cards found for {company_info.name} - this could be due to no employees or a loading issue')
				return profiles

			if not profile_cards:
				print(f'No profile cards found for {company_info.name}')
				return profiles

			for i, card in enumerate(profile_cards):
				try:
					# Extract name and LinkedIn URL using the correct selector
					name_link = await card.query_selector('a[data-test-app-aware-link]')
					if not name_link:
						html_content = await card.inner_html()
						print(f'No name link found for card {i}. Card HTML: {html_content}')
						continue

					# Try to get name from image alt first
					image = await name_link.query_selector('img')
					name = None
					if image:
						name = await image.get_attribute('alt')
						if name:
							# Remove " is open to work" if present
							name = name.replace(' is open to work', '')

					# If no name from image, try to get it from visually-hidden div
					if not name:
						ghost_name = await name_link.query_selector('div.visually-hidden')
						if ghost_name:
							name = await ghost_name.inner_text()

					if not name:
						html_content = await name_link.inner_html()
						print(f'No name found for card {i}. Link HTML: {html_content}')
						continue

					# Get the LinkedIn URL
					linkedin_url = await name_link.get_attribute('href')
					if not linkedin_url:
						print(f'No URL found for {name}')
						continue
					linkedin_url = linkedin_url.split('?')[0]

					# Extract short bio
					bio_element = await card.query_selector('div.t-14.t-black.t-normal')
					if not bio_element:
						html_content = await card.inner_html()
						print(f'No bio found for card {i}. Card HTML: {html_content}')
						continue
					bio = await bio_element.inner_text()
					bio = bio.strip()

					# Extract location
					location_element = await card.query_selector('div.t-14.t-normal')
					if not location_element:
						html_content = await card.inner_html()
						print(f'No location found for card {i}. Card HTML: {html_content}')
						continue
					location = await location_element.inner_text()
					location = location.strip()

					# Print complete profile information
					print(f'Found profile: {name} - {bio} at {location} ({linkedin_url})')

					profiles.append(
						{
							'name': name,
							'position': bio,  # Using bio as position since it contains the role
							'company': company_info.name,
							'linkedin_url': linkedin_url,
							'location': location,
						}
					)

					# Check if we've reached the maximum number of profiles
					if len(profiles) >= MAX_PROFILES:
						print(f'Reached maximum number of profiles ({MAX_PROFILES}) for {company_info.name}')
						return profiles

				except Exception as e:
					print(f'Error processing profile card: {str(e)}')

			# Check if there are more pages of results
			next_button = await page.query_selector('button[aria-label="Next"]')
			if not next_button:
				print('No more pages found')
				break

			# Check if the button is disabled by looking at both the class and attribute
			is_disabled = await next_button.get_attribute('disabled') is not None
			has_disabled_class = await next_button.evaluate('(element) => element.classList.contains("artdeco-button--disabled")')

			if is_disabled or has_disabled_class:
				print('Reached the last page')
				break

			# Click next button and wait for new page to load
			await next_button.click()
			await page.wait_for_load_state()
			await asyncio.sleep(2)

	except Exception as e:
		print(f'Error collecting profiles: {str(e)}')

	return profiles


async def filter_profiles(profiles: List[dict], llm: ChatDeepSeek) -> List[EmployeeInfo]:
	"""Filter profiles in batches to find relevant ones."""
	employee_profiles = []
	batch_size = 20

	# Process profiles in batches
	for i in range(0, len(profiles), batch_size):
		batch = profiles[i : i + batch_size]

		# Create a prompt for the LLM to evaluate the batch
		prompt = f"""
		Below is a list of positions. Please identify which ones are relevant to product management, marketing, sales, or executive positions (including CEO, CTO, COO, VP, Director, or any Chief positions).
		
		For each position, respond with its number if it is relevant, separated by commas.
		Example output: [1,3,5]
		Output the list of indices of the relevant positions only, no other text.
		Positions:
		{chr(10).join(f'{idx + 1}. {p["position"]}' for idx, p in enumerate(batch))}
		"""

		# Get LLM response
		response = llm.invoke(prompt)
		# Parse the response to get relevant indices
		relevant_indices = []
		match_indices = re.findall(r'\[(.*?)\]', response.content)
		if match_indices:
			for part in match_indices[0].split(','):
				try:
					idx = int(part.strip())
					relevant_indices.append(idx)
				except ValueError:
					continue

		# Add all profiles to the results, marking relevant ones
		for idx, profile in enumerate(batch, 1):
			employee_info = EmployeeInfo(
				name=profile['name'],
				position=profile['position'],
				company=profile['company'],
				linkedin_url=profile['linkedin_url'],
				relevancy=idx in relevant_indices,
			)
			employee_profiles.append(employee_info)
			if idx in relevant_indices:
				print(f'Found relevant employee: {profile["name"]} - {profile["position"]}')

	return employee_profiles


async def extract_employee_profiles(
	browser_context: BrowserContext, company_info: CompanyInfo, llm: ChatDeepSeek
) -> List[EmployeeInfo]:
	"""Extract profiles of employees with specific roles."""
	# First collect all profiles
	all_profiles = await collect_all_profiles(browser_context, company_info)
	print(f'Collected {len(all_profiles)} profiles from {company_info.name}')

	# Then filter them in batches
	return await filter_profiles(all_profiles, llm)


def load_existing_company_info() -> List[CompanyInfo]:
	"""Load existing company information from the CSV file."""
	companies = []
	if os.path.exists(COMPANY_INFO_FILE_PATH):
		df = pd.read_csv(COMPANY_INFO_FILE_PATH)
		for _, row in df.iterrows():
			companies.append(
				CompanyInfo(
					name=row['name'],
					id=str(row['id']) if pd.notna(row['id']) else None,  # Convert ID to string
					url=row['url'],
					description=row['description'],
					employee_count=row['employee_count'],
				)
			)
	return companies


async def process_company(
	browser_context: BrowserContext,
	company_info: CompanyInfo,
	llm: ChatDeepSeek,
	all_employee_profiles: List[EmployeeInfo],
	is_new_company: bool = False,
	force_search: bool = False,
) -> Optional[CompanyInfo]:
	"""Process a single company and update the employee profiles list."""
	try:
		# Skip if company is already completed and not in force search mode
		if not force_search and company_info.search_status == 'completed':
			print(f'Skipping {company_info.name} as it has already been processed')
			return company_info

		if is_new_company:
			# For new companies, we need to find their ID first
			found_company_info = await find_company_id(browser_context, company_info.name)
			if not found_company_info or not found_company_info.id:
				print(f'Could not find company info for {company_info.name}')
				return None
			company_info = found_company_info

		# Extract employee profiles
		employee_profiles = await extract_employee_profiles(browser_context, company_info, llm)
		all_employee_profiles.extend(employee_profiles)

		# Update company info with profile counts
		company_info.profiles_extracted = len(employee_profiles)
		company_info.relevant_profiles = sum(1 for profile in employee_profiles if profile.relevancy)
		company_info.search_status = 'completed'

		# Add a delay to avoid rate limiting
		await asyncio.sleep(3)
		return company_info
	except Exception as e:
		print(f'Error processing {"new" if is_new_company else "existing"} company {company_info.name}: {str(e)}')
		company_info.search_status = 'failed'
		return company_info


def save_all_csv(all_employee_profiles: List[EmployeeInfo], all_companies: List[CompanyInfo]):
	"""Save all employee profiles and companies to CSV."""
	if all_employee_profiles:
		df = pd.DataFrame(
			[
				{
					'name': profile.name,
					'position': profile.position,
					'company': profile.company,
					'linkedin_url': profile.linkedin_url,
					'relevancy': profile.relevancy,
				}
				for profile in all_employee_profiles
			]
		)
		df.to_csv(OUTPUT_FILE_PATH, index=False)
		print(f'Final save: {len(df)} employee profiles saved to {OUTPUT_FILE_PATH}')
	else:
		print('No employee profiles found')

	# Save all company info to CSV
	if all_companies:
		company_df = pd.DataFrame(
			[
				{
					'name': c.name,
					'id': c.id,
					'url': c.url,
					'description': c.description,
					'employee_count': c.employee_count,
					'profiles_extracted': c.profiles_extracted,
					'relevant_profiles': c.relevant_profiles,
					'search_status': c.search_status,
				}
				for c in all_companies
			]
		)
		company_df.to_csv(COMPANY_INFO_FILE_PATH, index=False)
		print(f'Final save: {len(company_df)} companies saved to {COMPANY_INFO_FILE_PATH}')
	else:
		print('No company info found')


async def main(force_search: bool = False):
	browser = Browser(
		config=BrowserConfig(
			browser_class='chromium',  # Must be set to chromium when using browser_binary_path
			browser_binary_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',  # macOS path
		)
	)

	# Initialize DeepSeek LLM with correct configuration
	llm = ChatDeepSeek(
		base_url='https://api.deepseek.com/v1',
		model='deepseek-chat',
		api_key=SecretStr(api_key),
	)

	try:
		# Create a browser context with your existing Chrome profile
		async with await browser.new_context(
			config=BrowserContextConfig(
				user_data_dir='~/Library/Application Support/Google/Chrome/Default',  # macOS path
				viewport={'width': 1920, 'height': 1080},  # Set viewport to maximize window
			)
		) as browser_context:
			# Maximize the window
			page = await browser_context.get_current_page()
			await page.set_viewport_size({'width': 1920, 'height': 1080})

			# Load existing company info
			existing_companies = load_existing_company_info()
			existing_company_names = {company.name for company in existing_companies}
			print(f'Loaded {len(existing_companies)} existing companies')

			# Get company names from the data file
			company_names = extract_company_names()
			new_company_names = [name for name in company_names if name not in existing_company_names]
			print(f'Found {len(new_company_names)} new companies to process')

			# Create a list to store the results
			all_employee_profiles = []
			all_companies = existing_companies.copy()

			# Process existing companies
			for company_info in existing_companies:
				updated_company_info = await process_company(
					browser_context, company_info, llm, all_employee_profiles, is_new_company=False, force_search=force_search
				)
				if updated_company_info:
					company_index = next((i for i, c in enumerate(all_companies) if c.name == updated_company_info.name), None)
					if company_index is not None:
						all_companies[company_index] = updated_company_info
				save_all_csv(all_employee_profiles, all_companies)

			# Process new companies
			for company_name in new_company_names:
				company_info = CompanyInfo(name=company_name)
				updated_company_info = await process_company(
					browser_context,
					company_info,
					llm,
					all_employee_profiles,
					is_new_company=True,
					force_search=force_search,
				)
				if updated_company_info:
					all_companies.append(updated_company_info)
				save_all_csv(all_employee_profiles, all_companies)
	finally:
		# Close the browser
		await browser.close()


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('--force', action='store_true', help='Force search for all companies, including those already processed')
	args = parser.parse_args()
	asyncio.run(main(force_search=args.force))
