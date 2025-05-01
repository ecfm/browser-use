# Goal: Search LinkedIn for companies and extract employee profiles for product manager, marketing, sales, and executive roles.

import asyncio
import os
import random
import re
from typing import Dict, List, Optional

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
	school_name: Optional[str] = None
	company_category: Optional[str] = None
	location: Optional[str] = None


UIUC_SCHOOL_ID = '2650'
CMU_SCHOOL_ID = '3147'
MIT_SCHOOL_ID = '1503'

SCHOOLS: Dict[str, str] = {
	'UIUC': '2650',
	'CMU': '3147',
	'MIT': '1503',
	# Add more schools here if needed: 'School Name': 'LinkedIn School ID'
}

# Constants
# Maximum number of profiles to collect per company *per school search*
# Note: The optimization logic uses this value differently.
MAX_PROFILES = 80
DATA_NAMES = ['tools_and_home_improvement']  # , 'appliances', 'electronics']
COMPANY_SEARCH_LINK = 'https://www.linkedin.com/search/results/companies/?keywords='
PEOPLE_SEARCH_LINK_TEMPLATE = 'https://www.linkedin.com/search/results/people/?currentCompany=%5B%22{company_id}%22%5D&origin=FACETED_SEARCH&schoolFilter=%5B%22{school_id}%22%5D&sid=profile_school_filter'

# Define the single output file for all profiles
COMBINED_PROFILES_OUTPUT_PATH = 'results/all_categories_by_school_linkedin_profiles.csv'


async def collect_all_profiles(
	browser_context: BrowserContext,
	company_info: CompanyInfo,
	school_id: str,
	school_name: str,
	company_category: str,  # Added category
) -> List[dict]:
	"""Collect all employee profiles from a company for a specific school filter."""
	profiles = []
	processed_urls = set()  # Keep track of processed URLs to avoid duplicates
	# MAX_PROFILES is now defined globally

	if not company_info.id:
		return profiles

	# Navigate to the people search page using the school ID
	people_search_url = PEOPLE_SEARCH_LINK_TEMPLATE.format(company_id=company_info.id, school_id=school_id)
	page = await browser_context.get_current_page()
	await page.goto(people_search_url)
	await page.wait_for_load_state()
	print(f'Searching for employees at {company_info.name} for school: {school_name}')

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
					print(f'No employee profiles found for {company_info.name} for school: {school_name}')
					return profiles

			# Find profile cards using data attributes
			try:
				await page.wait_for_selector('div[data-chameleon-result-urn*="urn:li:member"]', timeout=10000)
				profile_cards = await page.query_selector_all('div[data-chameleon-result-urn*="urn:li:member"]')
			except Exception as e:
				print(f'No profile cards found for {company_info.name} - this could be due to no employees or a loading issue')
				return profiles

			if not profile_cards:
				print(f'No profile cards found for {company_info.name} for school: {school_name}')
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

					# Extract short bio - Target the specific class combo
					bio_selector = 'div.t-14.t-black.t-normal'
					bio_element = await card.query_selector(bio_selector)

					if not bio_element:
						html_content = await card.inner_html()
						print(f'No bio found for card {i} using selector {bio_selector}. Card HTML: {html_content}')
						bio = '[Bio N/A]'  # Set default if not found
					else:
						bio = await bio_element.inner_text()
						bio = bio.strip()

					# Extract location - Find all elements with general location classes
					potential_location_selector = 'div.t-14.t-normal'
					potential_location_elements = await card.query_selector_all(potential_location_selector)

					location_element = None
					location = '[Location N/A]'  # Default value
					for potential_loc in potential_location_elements:
						try:
							potential_loc_text = await potential_loc.inner_text()
							potential_loc_text = potential_loc_text.strip()
							# Compare text content, ensuring bio is not N/A
							if potential_loc_text and bio != '[Bio N/A]' and potential_loc_text != bio:
								location = potential_loc_text
								break  # Found a distinct location based on text
							# Handle case where bio element wasn't found
							elif potential_loc_text and bio == '[Bio N/A]':
								# If bio is unknown, take the first non-empty potential location
								location = potential_loc_text
								break
						except Exception as text_error:
							print(f'Error getting text from potential location element: {text_error}')
							continue

					# If loop finishes without finding a distinct location text, 'location' remains '[Location N/A]'

					# Print complete profile information
					# print(f'Found profile: {name} - {bio} at {location} ({linkedin_url})') # Commented out to reduce noise

					profiles.append(
						{
							'name': name,
							'position': bio,  # Using bio as position since it contains the role
							'company': company_info.name,
							'linkedin_url': linkedin_url,
							'location': location,
							'school_name': school_name,
							'company_category': company_category,  # Store category
						}
					)

					# Check if we've reached the maximum number of profiles
					if len(profiles) >= MAX_PROFILES:
						print(
							f'Reached maximum number of profiles ({MAX_PROFILES}) for {company_info.name} for school: {school_name}'
						)
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


async def filter_profiles(
	profiles: List[dict], llm: ChatDeepSeek
) -> List[EmployeeInfo]:  # Removed company_name and company_category params
	"""Filter profiles in batches to find relevant ones."""
	employee_profiles = []
	batch_size = 20

	# Process profiles in batches
	for i in range(0, len(profiles), batch_size):
		batch = profiles[i : i + batch_size]

		# Create a prompt for the LLM to evaluate the batch
		prompt = f"""
		I'm looking for connections in the USA who will likely be the end-user or key decision-makers of adopting a new market research and customer insights tool.
		Below is a list of profiles. Please identify which ones are relevant, e.g. product management, marketing, sales, or executive positions that are responsible for or interested in these areas. Also consider the relevance of market research and customer insights to the company they work for.
		
		For each profile, respond with its number if it is relevant, separated by commas.
		Example output: [1,3,5]
		Output the list of indices of the relevant profile in the USA only, no other text.
		Profiles:
		{chr(10).join(f'{idx + 1}. {p["position"]} at {p["company"]} in {p["location"]}' for idx, p in enumerate(batch))}
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
				school_name=profile['school_name'],
				company_category=profile.get('company_category'),  # Get category from profile dict
				location=profile.get('location', '[Location N/A]'),
			)
			employee_profiles.append(employee_info)
			if idx in relevant_indices:
				print(f'Found relevant employee: {profile["name"]} - {profile["position"]}')

	return employee_profiles


async def extract_employee_profiles(
	browser_context: BrowserContext,
	company_info: CompanyInfo,
	llm: ChatDeepSeek,
	school_id: str,
	school_name: str,
	company_category: str,
) -> List[EmployeeInfo]:
	"""Extract profiles of employees with specific roles for a given school filter."""
	# First collect all profiles for the specific school
	all_profiles = await collect_all_profiles(browser_context, company_info, school_id, school_name, company_category)
	print(f'Collected {len(all_profiles)} profiles from {company_info.name} for school: {school_name}')

	# This function is no longer the primary path, but filtering call retained for potential reuse
	# If called, it would filter just the profiles from this specific school/company combo
	return await filter_profiles(all_profiles, llm)


def load_existing_company_info(company_info_file_path: str) -> List[CompanyInfo]:
	"""Load existing company information from the CSV file."""
	companies = []
	if not os.path.exists(company_info_file_path):
		print(f'Warning: Company info file not found: {company_info_file_path}. Skipping category.')
		return companies

	try:
		df = pd.read_csv(company_info_file_path)
		# Ensure relevant_profiles column exists and handle potential NaN
		if 'relevant_profiles' not in df.columns:
			raise ValueError(f"Missing 'relevant_profiles' column in {company_info_file_path}")
		df['relevant_profiles'] = pd.to_numeric(df['relevant_profiles'], errors='coerce').fillna(0).astype(int)
		# Ensure profiles_extracted column exists and handle potential NaN
		if 'profiles_extracted' not in df.columns:
			print(f"Warning: Missing 'profiles_extracted' column in {company_info_file_path}. Defaulting to 0.")
			df['profiles_extracted'] = 0  # Add column with default if missing
		df['profiles_extracted'] = pd.to_numeric(df['profiles_extracted'], errors='coerce').fillna(0).astype(int)

		for _, row in df.iterrows():
			# Ensure ID is treated as string, handling potential floats from CSV read
			company_id_str = None
			if pd.notna(row['id']):
				try:
					# Convert to int first to remove decimal, then to string
					company_id_str = str(int(row['id']))
				except ValueError:
					# Handle cases where ID might not be a simple number (though less likely)
					company_id_str = str(row['id'])

			companies.append(
				CompanyInfo(
					name=row['name'],
					id=company_id_str,
					url=row['url'],
					description=row['description'],
					employee_count=row['employee_count'],
					relevant_profiles=row['relevant_profiles'],
					search_status=row.get('search_status', 'pending'),
					profiles_extracted=row['profiles_extracted'],  # Load profiles_extracted
				)
			)
	except Exception as e:
		print(f'Error loading company info from {company_info_file_path}: {e}')

	return companies


def save_all_csv(
	all_employee_profiles: List[EmployeeInfo],
	output_profiles_path: str,
):
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
					'school_name': profile.school_name,
					'company_category': profile.company_category,
					'location': profile.location,
				}
				for profile in all_employee_profiles
			]
		)
		try:
			os.makedirs(os.path.dirname(output_profiles_path), exist_ok=True)
			df.to_csv(output_profiles_path, index=False)
			print(f'Saved {len(df)} employee profiles to {output_profiles_path}')
		except Exception as e:
			print(f'Error saving employee profiles to {output_profiles_path}: {e}')
	else:
		print('No employee profiles found to save.')


async def main():
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
			page = await browser_context.get_current_page()
			await page.set_viewport_size({'width': 1920, 'height': 1080})

			# --- Initialize list for all profiles across all categories --- (Now stores dicts)
			master_employee_profile_list: List[dict] = []

			# --- Loop through each data category ---
			for data_name in DATA_NAMES:
				print(f'\n{"=" * 20} Processing Category: {data_name.upper()} {"=" * 20}')

				# --- Define file paths for the current category ---
				original_company_info_file_path = f'results/{data_name}_linkedin_companies.csv'
				# output_profiles_path = f'results/{data_name}_by_school_linkedin_profiles.csv' # This path seems unused now

				# Load existing company info for this category
				all_loaded_companies = load_existing_company_info(original_company_info_file_path)
				if not all_loaded_companies:
					# Already handled within load_existing_company_info, so just continue to next category
					continue
				print(f'Loaded {len(all_loaded_companies)} total companies from {original_company_info_file_path}')

				# Filter companies: must have an ID and previously found relevant profiles > 0
				companies_to_process = [c for c in all_loaded_companies if c.id and c.relevant_profiles > 0]
				print(
					f'Found {len(companies_to_process)} companies with ID and >0 relevant profiles to process for school search.'
				)

				if not companies_to_process:
					print('No companies meet the criteria for school-specific searching. Exiting.')
					continue  # Skip category if no companies meet criteria

				# updated_company_info_list = all_loaded_companies.copy()

				# Process the filtered companies
				for company_info in companies_to_process:
					print(f'\n=== Processing Company: {company_info.name} (ID: {company_info.id}) ===')

					# --- NEW: List to hold all profiles for this company across all schools --- (Still used temporarily)
					company_all_schools_profiles = []

					# Iterate through each school for the current company
					for school_name, school_id in SCHOOLS.items():
						print(f'--- Collecting for School: {school_name} ({school_id}) ---')
						# Directly collect profiles, don't filter yet
						raw_profiles = await collect_all_profiles(
							browser_context,
							company_info,
							school_id,  # Pass correct school ID
							school_name,  # Pass correct school Name
							data_name,  # Pass category
						)
						if raw_profiles:
							# profiles_collected_this_company += len(raw_profiles) # Track count if needed later
							company_all_schools_profiles.extend(raw_profiles)
						else:
							print(f'No profiles collected for {school_name}.')

						# Add a small delay between school searches for the same company
						await asyncio.sleep(random.uniform(2, 5))

					# --- MOVED: Filtering is now done after collecting ALL profiles ---
					if company_all_schools_profiles:
						print(
							f'Adding {len(company_all_schools_profiles)} collected profiles from {company_info.name} to master list.'
						)
						master_employee_profile_list.extend(company_all_schools_profiles)
					else:
						print(f'No profiles collected for {company_info.name} across any school for this category.')

					# Add a delay between companies
					await asyncio.sleep(random.uniform(5, 10))

				# --- End of Category Processing --- #
				print(f'\n{"=" * 20} Finished Category: {data_name.upper()} {"=" * 20}')

			# --- NEW: Filter the entire master list after all collection is done ---
			print(f'\nFiltering {len(master_employee_profile_list)} total collected profiles across all categories...')
			if master_employee_profile_list:
				final_filtered_list = await filter_profiles(master_employee_profile_list, llm)
				print(f'Filtered down to {len(final_filtered_list)} relevant profiles.')
			else:
				final_filtered_list = []
				print('No profiles collected across all categories to filter.')

			# --- Load existing profiles from the combined file --- #
			print(f'\nLoading existing profiles from {COMBINED_PROFILES_OUTPUT_PATH}...')
			existing_profiles_list: List[EmployeeInfo] = []
			if os.path.exists(COMBINED_PROFILES_OUTPUT_PATH):
				try:
					existing_df = pd.read_csv(COMBINED_PROFILES_OUTPUT_PATH)
					# Convert existing DataFrame rows to EmployeeInfo objects
					for _, row in existing_df.iterrows():
						# Handle potential NaN values, especially for optional fields
						existing_profiles_list.append(
							EmployeeInfo(
								name=row.get('name', '[Name N/A]'),
								position=row.get('position', '[Position N/A]'),
								company=row.get('company', '[Company N/A]'),
								linkedin_url=row.get('linkedin_url'),  # URL is crucial, hope it exists
								relevancy=bool(row.get('relevancy', False)),
								school_name=row.get('school_name') if pd.notna(row.get('school_name')) else None,
								company_category=row.get('company_category') if pd.notna(row.get('company_category')) else None,
								location=row.get('location', '[Location N/A]'),
							)
						)
					print(f'Loaded {len(existing_profiles_list)} existing profiles.')
				except Exception as e:
					print(
						f'Error loading or processing existing profiles file {COMBINED_PROFILES_OUTPUT_PATH}: {e}. Proceeding with only new profiles.'
					)
					existing_profiles_list = []  # Reset if error occurred
			else:
				print('No existing combined profile file found. Starting fresh.')

			# --- Combine newly filtered profiles with existing ones --- #
			all_profiles_to_process = final_filtered_list + existing_profiles_list
			print(f'Total profiles to deduplicate (new + existing): {len(all_profiles_to_process)}')

			# --- Final Save after all categories are processed --- #
			# --- Deduplicate and Merge School Names --- #
			print(f'\nProcessing {len(all_profiles_to_process)} total profiles for deduplication...')  # Updated count message
			unique_profiles_dict: Dict[str, EmployeeInfo] = {}

			for profile_info in all_profiles_to_process:  # Use the combined list
				if not profile_info.linkedin_url:  # Skip if URL is missing
					print(f'Skipping profile due to missing URL: {profile_info.name}')
					continue

				url = profile_info.linkedin_url
				current_school = profile_info.school_name

				if url in unique_profiles_dict:
					# Profile already exists, merge school name and update relevancy
					existing_profile = unique_profiles_dict[url]

					# Merge school name
					existing_schools = existing_profile.school_name
					if existing_schools and current_school:
						# Split existing schools into a set for easy checking
						school_set = set(s.strip() for s in existing_schools.split(',') if s.strip())
						if current_school not in school_set:
							existing_profile.school_name = f'{existing_schools}, {current_school}'
					elif not existing_schools and current_school:
						# Handle case where existing school was None/empty
						existing_profile.school_name = current_school

					# Update relevancy (if relevant in any search, mark as relevant)
					existing_profile.relevancy = existing_profile.relevancy or profile_info.relevancy

					# Optional: Update category if needed (e.g., comma-separate or keep first)
					# existing_category = existing_profile.company_category
					# current_category = profile_info.company_category
					# if existing_category and current_category and current_category not in existing_category.split(','):
					# 	existing_profile.company_category = f"{existing_category}, {current_category}"
					# elif not existing_category and current_category:
					# 	existing_profile.company_category = current_category

				else:
					# New profile, add to dictionary
					unique_profiles_dict[url] = profile_info

			final_profile_list = list(unique_profiles_dict.values())
			print(f'Saving {len(final_profile_list)} unique profiles after merging.')
			save_all_csv(final_profile_list, COMBINED_PROFILES_OUTPUT_PATH)

	finally:
		# Close the browser after processing all categories
		await browser.close()
		print('\nBrowser closed.')


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	# No arguments needed anymore
	# args = parser.parse_args()
	asyncio.run(main())
