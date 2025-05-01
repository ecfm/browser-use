import argparse
import asyncio
import os
import random
import re  # Need re for substitution
import time
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from browser_use import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig

# --- Environment & LLM Setup ---
load_dotenv()
# Load both potential keys
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

# We will check for the required key based on user choice later
# if not api_key:
# 	# Allow running connect mode without API key, but raise error in draft mode if missing
# 	pass
# raise ValueError('DEEPSEEK_API_KEY is not set in the environment variables. Needed for draft mode.')

# --- Configuration ---
# Removed SCHOOL_NAME and MESSAGE_TEMPLATE
INPUT_PROFILES_CSV_DEFAULT = 'results/linkedin_connection_messages.csv'
MESSAGES_CSV_DEFAULT = 'results/linkedin_connection_messages.csv'

# Rate limiting - Adjust as needed (values in seconds)
MIN_DELAY_BETWEEN_REQUESTS = 1  # Minimum delay between connection requests
MAX_DELAY_BETWEEN_REQUESTS = 2  # Maximum delay between connection requests
PAGE_LOAD_DELAY = 2  # Increased delay for page navigation/loading elements

# Retryable statuses for connect mode
RETRYABLE_STATUSES = [
	'pending',
	'failed - navigation error',
	'failed - connect button not found',
	'failed - modal not found',
	'failed - add note error',
	'failed - message text area not found',
	'failed - message fill error',
	'failed - send button not found',
	'failed - send error',
	'failed - send button disabled',
	'failed - unexpected error',  # Allow retrying generic errors
]


# Pydantic model for connection message data
class ConnectionMessage(BaseModel):
	name: str
	position: str
	company: str
	linkedin_url: str
	relevancy: bool
	school_name: Optional[str] = None
	company_category: Optional[str] = None
	location: Optional[str] = None
	generated_message: Optional[str] = None  # Store the LLM generated message
	connection_status: str = 'pending'  # Track status


# --- Helper Functions ---


async def print_debug_html(page, failure_context: str, max_chars: int = 20000):
	"""Helper function to print cleaned HTML for debugging."""
	try:
		html = await page.content()

		# Step 1: Collapse consecutive meta/link tags
		html = re.sub(
			r'((?:<(?:meta|link)[^>]*>\s*){2,})', '<!-- Multiple meta/link tags collapsed -->\\n', html, flags=re.DOTALL
		)

		# Step 2: Collapse consecutive script tags
		html = re.sub(
			r'((?:<script[^>]*>.*?</script>\s*){2,})', '<!-- Multiple script tags collapsed -->\\n', html, flags=re.DOTALL
		)

		# Step 3: Collapse consecutive svg tags
		html = re.sub(r'((?:<svg[^>]*>.*?</svg>\s*){2,})', '<!-- Multiple svg tags collapsed -->\\n', html, flags=re.DOTALL)

		# Step 4: Collapse consecutive hidden code tags
		html = re.sub(
			r'((?:<code[^>]*style="display: none"[^>]*>.*?</code\s*>){2,})',
			'<!-- Multiple hidden code tags collapsed -->\\n',
			html,
			flags=re.DOTALL,
		)

		# Step 5: Shorten long meta content (only if not already collapsed)
		html = re.sub(
			r'(<meta(?!.*<!-- Multiple meta/link tags collapsed -->)[^>]*content=")([^"]*)("[^>]*>)',
			lambda m: m.group(1) + ('[...long content...]' if len(m.group(2)) > 50 else m.group(2)) + m.group(3),
			html,
		)

		# Step 6: Shorten individual remaining SVG content
		html = re.sub(
			r'(<svg(?!.*<!-- Multiple svg tags collapsed -->)[^>]*>).*?(</svg>)',
			r'\1[...]',  # Intentionally leave off \2 to only keep opening tag + placeholder
			html,
			flags=re.DOTALL,
		)

		# Step 7: Shorten individual remaining hidden code content
		html = re.sub(
			r'(<code(?!.*<!-- Multiple hidden code tags collapsed -->)[^>]*style="display: none"[^>]*>).*?(</code\s*>)',
			r'\1[...hidden data...]\2',
			html,
			flags=re.DOTALL,
		)

		# Step 8: Remove hidden tracking images
		html = re.sub(r'<img src="data:image/gif.*?" style="display: none".*?>', '', html)

		# Step 9: Remove specific framework comments
		html = re.sub(r'<!-- EMBER_CLI_FASTBOOT.*?-->', '', html, flags=re.DOTALL)
		html = re.sub(r'<!--%SSR_HEAD.*?%-->', '', html, flags=re.DOTALL)

		# Step 10: Shorten inline style blocks
		html = re.sub(r'(<style[^>]*>).*?(</style>)', r'\1/* ... styles ... */\2', html, flags=re.DOTALL)

		# Step 11: Remove general HTML comments
		html = re.sub(r'<!--(?! Multiple | \d+ / \d+ ).*?-->', '', html, flags=re.DOTALL)

		print(f'\\n--- Relevant HTML ({failure_context}) ---')
		print(html[:max_chars])
		print('--- End Relevant HTML ---\\n')
	except Exception as html_e:
		print(f'Could not get or clean HTML content for debugging ({failure_context}): {html_e}')


def load_input_profiles(file_path: str) -> pd.DataFrame:
	"""Loads profiles from the specified CSV file generated by the search script."""
	if not os.path.exists(file_path):
		raise FileNotFoundError(f'Input profiles CSV file not found: {file_path}')
	df = pd.read_csv(file_path)
	# Ensure relevancy column is boolean
	if 'relevancy' not in df.columns:
		raise ValueError("Input profiles CSV must contain a 'relevancy' column.")
	# Attempt to convert, handling potential non-boolean strings if needed
	try:
		# Convert potential strings 'True'/'False' to boolean
		df['relevancy'] = df['relevancy'].apply(
			lambda x: str(x).strip().lower() == 'true' if isinstance(x, (str, bool)) else False
		)
	except Exception as e:
		print(f"Warning: Could not reliably convert 'relevancy' column to boolean: {e}. Assuming False for errors.")
		df['relevancy'] = False  # Default to False on conversion error

	# Ensure essential columns exist
	required_cols = ['name', 'position', 'company', 'linkedin_url', 'relevancy', 'school_name', 'company_category', 'location']
	for col in required_cols:
		if col not in df.columns:
			print(f"Warning: Input profiles CSV missing expected column '{col}'. It will be treated as empty/None.")
			df[col] = None  # Add missing column with None

	return df


def load_connection_messages(file_path: str) -> pd.DataFrame:
	"""Loads existing connection messages from the specified CSV file."""
	if not os.path.exists(file_path):
		print(f'Messages CSV file not found: {file_path}. Will create a new one.')
		# Return empty DataFrame with expected columns if file doesn't exist
		return pd.DataFrame(columns=ConnectionMessage.__annotations__.keys())
	try:
		df = pd.read_csv(file_path)
		# Ensure all columns from the model exist, add if missing
		for col in ConnectionMessage.__annotations__.keys():
			if col not in df.columns:
				print(f"Adding missing column '{col}' to existing messages CSV.")
				df[col] = None  # Add missing column
		# Fill NaNs in status with 'pending' for safety
		if 'connection_status' in df.columns:
			df['connection_status'] = df['connection_status'].fillna('pending')
		else:
			df['connection_status'] = 'pending'
		# Fill NaNs in relevancy with False for safety
		if 'relevancy' in df.columns:
			df['relevancy'] = df['relevancy'].fillna(False).astype(bool)
		else:
			df['relevancy'] = False
		# Fill NaNs in generated_message with None
		if 'generated_message' in df.columns:
			df['generated_message'] = df['generated_message'].fillna(value=pd.NA)  # Use pd.NA for consistency
		else:
			df['generated_message'] = pd.NA

		return df
	except Exception as e:
		print(f"Error loading messages CSV '{file_path}': {e}. Returning empty DataFrame.")
		return pd.DataFrame(columns=ConnectionMessage.__annotations__.keys())


def save_connection_messages(df: pd.DataFrame, file_path: str):
	"""Saves the DataFrame with connection messages and statuses to CSV."""
	try:
		# Ensure directory exists
		os.makedirs(os.path.dirname(file_path), exist_ok=True)
		df.to_csv(file_path, index=False)
		print(f'Connection messages and statuses saved to {file_path}')
	except Exception as e:
		print(f"Error saving connection messages to '{file_path}': {e}")


# Removed get_first_name and customize_message


async def generate_message_with_llm(profile: pd.Series, llm: ChatDeepSeek, school_name: Optional[str]) -> Optional[str]:
	"""Generates a personalized connection message using the LLM."""
	position = profile.get('position', '[Position N/A]')
	company = profile.get('company', '[Company N/A]')

	# Handle potentially missing or multi-school names
	school_phrase = ''  # Default empty
	if pd.notna(school_name) and school_name.strip():
		school_list = [s.strip() for s in school_name.split(',') if s.strip()]
		school_phrase = f'({"&".join(school_list)})'
		# If school_name was whitespace or empty, school_phrase remains empty
	else:
		school_phrase = ''  # Generic fallback if no school listed

	prompt = f"""
    Please craft a concise and professional LinkedIn connection request message (less than 300 characters).
    The recipient's name is {profile['name']}.
    Their role is {position} at {company}.
    We are also an alum from the same school(s) {school_phrase}.
    I'm a PhD at MIT Sloan building AI tools for extracting insights (e.g. customer needs, product usages, performance of specific product features, etc.) from large volume of unstructured data (e.g. online reviews, social media, support tickets, etc.). Help me to customize the message according to the recipient's background and mutual interest in the technology. The description of my research can be narrowed down or broadened based on the recipient's background and interests.
    Keep the tone professional and friendly. Avoid generic phrases.
    Do not include placeholders like <name>. Output only the message text.

    Follow the exact format below:
    Hi <name>, saw we're both <school> alums! As a PhD at MIT Sloan developing AI tools for <a concise phrase describing my research with optional short detail if that helps to match the recepient's background or interests>, I admire your <a phrase describing the recipient's background>. Would love to connect.

    Generate the message now and nothing else.
    """
	try:
		response = llm.invoke(prompt)
		message = response.content.strip()
		# Basic validation
		if len(message) > 300:
			print(f'Warning: Generated message for {profile["name"]} is > 300 chars. Truncating.')
			message = message[:297] + '...'
		if not message:
			print(f'Warning: LLM returned empty message for {profile["name"]}.')
			return None
		return message
	except Exception as e:
		print(f'Error generating message for {profile["name"]} using LLM: {e}')
		return None


async def send_connection_request(browser_context: BrowserContext, profile_url: str, message: str) -> str:
	"""Navigates to a profile and sends a connection request with a message."""
	# --- Function body remains largely the same as before ---
	# Ensure it handles potential None message gracefully if needed, although connect mode should only process profiles with messages.
	if not message:
		print('Error: Attempting to send connection without a message.')
		return 'failed - no message provided'

	page = await browser_context.get_current_page()
	status = 'failed - navigation error'
	try:
		print(f'Navigating to profile: {profile_url}')
		await page.goto(profile_url, wait_until='domcontentloaded')  # Wait for DOM content
		await asyncio.sleep(PAGE_LOAD_DELAY)  # Allow dynamic content to load (using updated delay)
		body_html = await page.content()
		# save the html to a file
		# with open('profile_page.html', 'w') as f: # Commented out saving
		#     f.write(body_html)

		# Check if already connected or pending first
		pending_button = await page.query_selector('button:has-text("Pending")')  # Keep this selector
		if pending_button and await pending_button.is_visible():
			print('Connection status is Pending.')
			return 'pending - already sent'

		# Add check for "Message" button indicating connection exists
		# Try finding a message button within the main profile card area as an indicator of connection
		# Using a selector that is likely to contain the main action buttons
		main_card_action_area_selector = "section.artdeco-card div:has(> button[aria-label*='Message'])"

		# --- Start NEW Button Finding Logic ---
		connect_button = None
		is_in_more_dropdown = False
		# Define a scope for the main profile card which usually contains the H1 name
		# Using :has(h1) makes it slightly more specific
		main_profile_card_selector = 'section.artdeco-card:has(h1)'

		# Priority 1: Look for direct "Connect" button within the main profile card
		direct_connect_selectors = [
			f'{main_profile_card_selector} button:has-text("Connect"):not([aria-label*="message"])',  # Text-based
			f'{main_profile_card_selector} button[aria-label*="Invite"][aria-label*="to connect"]',  # Aria-label based
		]
		print(f'Searching for direct Connect button within main profile card ({main_profile_card_selector})...')
		for selector in direct_connect_selectors:
			try:
				print(f'  Trying direct connect selector: {selector}')
				# Use page.query_selector directly
				connect_button_found = await page.query_selector(selector)
				if connect_button_found and await connect_button_found.is_visible():
					print(f'    Found direct connect button using selector: {selector}')
					connect_button = connect_button_found  # Assign if found and visible
					break  # Found it
				else:
					print(f'    Selector {selector} did not find a visible button.')
					# connect_button remains None or previous value
			except Exception as e:
				print(f"    Direct connect selector '{selector}' search failed: {e}")
				# connect_button remains None or previous value
			if connect_button:
				break  # Exit loop if button found

		# Priority 2: If no direct connect, look for "More" button within the main profile card
		if not connect_button:
			print(
				f'Direct connect button not found, searching for "More" button within main profile card ({main_profile_card_selector})...'
			)
			# Selectors for the "More" button within the card
			more_button_selectors = [
				f'{main_profile_card_selector} button[aria-label="More actions"]',  # Specific aria-label
				f'{main_profile_card_selector} button:has-text("More")',  # Text based
			]
			more_button = None
			for selector in more_button_selectors:
				try:
					print(f'  Trying More button selector: {selector}')
					# Use page.query_selector directly
					more_button_found = await page.query_selector(selector)
					if more_button_found and await more_button_found.is_visible():
						print(f'    Found "More" button using selector: {selector}')
						more_button = more_button_found  # Assign if found and visible
						break  # Found it
					else:
						print(f'    Selector {selector} did not find a visible button.')
						# more_button remains None or previous value
				except Exception as e:
					print(f"    More button selector '{selector}' search failed: {e}")
					# more_button remains None or previous value
				if more_button:
					break  # Exit loop if button found

			# If "More" button found, click it and find "Connect" in dropdown (page search)
			if more_button:
				try:
					print("Clicking 'More' button...")
					await more_button.click()
					await asyncio.sleep(1.5)  # Slightly longer wait for dropdown

					# --- Refined Dropdown Search ---
					print('Searching for the dropdown container...')
					# Selector for the visible dropdown content area
					dropdown_container_selector = 'div.artdeco-dropdown__content.artdeco-dropdown__content--is-open'
					dropdown_container = await page.query_selector(dropdown_container_selector)

					if dropdown_container and await dropdown_container.is_visible():
						print("Dropdown container found. Searching for 'Connect' within it...")
						# Selectors to use *within* the container
						# Note: These are relative selectors now, starting with '.' or tag name
						dropdown_connect_selectors_relative = [
							'.artdeco-dropdown__item:has(svg[data-test-icon="connect-medium"])',  # Icon-based
							'.artdeco-dropdown__item:has-text("Connect")',  # Text-based
							'.artdeco-dropdown__item[aria-label*="to connect"]',  # Aria-label based
						]
						for dd_selector in dropdown_connect_selectors_relative:
							try:
								print(f'  Trying relative selector in dropdown: {dd_selector}')
								# Search within the found container
								connect_button_in_dropdown = await dropdown_container.query_selector(dd_selector)
								if connect_button_in_dropdown and await connect_button_in_dropdown.is_visible():
									print(f"    Found 'Connect' in dropdown using relative selector: {dd_selector}")
									connect_button = connect_button_in_dropdown  # Assign the dropdown button
									is_in_more_dropdown = True
									break  # Found connect button in dropdown
								else:
									print(f'    Relative selector {dd_selector} did not find a visible button in dropdown.')
							except Exception as e:
								print(f"    Dropdown relative connect selector '{dd_selector}' search failed: {e}")
							# connect_button remains None or previous value
						# Exit inner loop if button found
						if connect_button:
							pass  # connect_button is now assigned, loop will finish or next outer check handles it

						# Check if connect button was found within the container
						if not connect_button:
							print("'Connect' button not found within the identified dropdown container.")
							# Print HTML of the container for debugging
							try:
								dropdown_html = await dropdown_container.inner_html()
								print('\\n--- Relevant HTML (Dropdown Container Content) ---')
								print(dropdown_html[:2000])
								print('--- End Relevant HTML ---\\n')
							except Exception as html_e:
								print(f'Could not get dropdown container HTML: {html_e}')
					else:
						print("Dropdown container not found or not visible after clicking 'More'.")
						# Optional: print wider area HTML if container is missing
						try:
							await print_debug_html(page, 'Dropdown Container Not Found', 5000)
						except Exception as dbg_html_e:
							print(f'Error printing debug HTML: {dbg_html_e}')
					# --- End Refined Dropdown Search ---
				except Exception as e:
					print(f"Error clicking 'More' button or searching dropdown: {e}")
					connect_button = None  # Ensure connect_button is None if More click or dropdown search failed
			# Else (no More button found)
			else:
				print('Neither direct Connect nor More button found within the main profile card.')
				# connect_button is already None if we get here

		# --- End NEW Button Finding Logic ---

		# --- Proceed with checks and clicking ---
		if not connect_button:
			# Re-check pending/connected status - Already done earlier

			print('Could not find Connect button via direct search or More dropdown within the main profile card.')
			# --- Added HTML printing ---
			try:
				# Print cleaned page HTML as a fallback
				print('Printing cleaned page HTML for debugging lack of Connect/More button...')
				await print_debug_html(page, 'Connect Button Not Found - Full Body Fallback', 10000)

			except Exception as html_e:
				print(f'Could not get HTML content for debugging: {html_e}')
			# --- End Added HTML printing ---
			await page.screenshot(path=f'screenshots/connect_fail_{time.time()}.png')
			return 'failed - connect button not found'

		# Click the found connect button (whether direct or from dropdown)
		try:
			await connect_button.click()
			print('Clicked Connect button.')
			await asyncio.sleep(PAGE_LOAD_DELAY)  # Wait for modal
		except Exception as e:
			print(f'Error clicking the found connect button: {e}')
			# --- Use Helper for HTML Print ---
			await print_debug_html(page, 'Click Connect Error', 10000)
			# --- End ---
			await page.screenshot(path=f'screenshots/click_connect_fail_{time.time()}.png')
			return 'failed - clicking connect button error'

		# --- Step 2: Add a Note ---
		add_note_button = None
		modal = None
		try:
			# Selector for the modal dialog
			modal_selector = 'div[role="dialog"][aria-labelledby*="send-invite"]'
			await page.wait_for_selector(modal_selector, timeout=1000)  # Increased timeout slightly
			modal = await page.query_selector(modal_selector)
			if modal:
				add_note_selector = 'button[aria-label="Add a note"], button:has-text("Add note")'  # Updated selector
				add_note_button = await modal.query_selector(add_note_selector)

				# Look for alternative if primary not found
				if not add_note_button or not await add_note_button.is_visible():
					add_note_selector_alt = 'button:has-text("Personalize invitation")'  # Another common text
					add_note_button = await modal.query_selector(add_note_selector_alt)

				if add_note_button and await add_note_button.is_visible():
					await add_note_button.click()
					print("Clicked 'Add a note' / 'Personalize' button.")
					await asyncio.sleep(1.5)  # Wait for text area to appear reliably
				else:
					print("'Add a note' button not found in modal. Trying to send without note.")
					# Attempt to send directly if no note option is easily found
					send_now_button_no_note = await modal.query_selector(
						'button[aria-label="Send now"], button:has-text("Send")'
					)  # Broader send button
					if send_now_button_no_note and await send_now_button_no_note.is_visible():
						if not await send_now_button_no_note.is_disabled():
							await send_now_button_no_note.click()
							print('Sent connection request without a note (modal appeared, no note button).')
							return 'success - no note'
						else:
							print("Direct 'Send' button found but disabled.")
							return 'failed - send button disabled (no note)'
					else:
						print("Could not find 'Send now' button in modal when 'Add note' was missing.")
						# --- Added HTML printing ---
						try:
							modal_html_content = await modal.inner_html()
							print('\\n--- Relevant HTML (Modal - No Send Button without Note) ---')
							print(modal_html_content[:3000])  # Increased snippet size
							print('--- End Relevant HTML ---\\n')
						except Exception as html_e:
							print(f'Could not get modal HTML content for debugging: {html_e}')
						# --- End Added HTML printing ---
						await page.screenshot(path=f'screenshots/modal_no_buttons_{time.time()}.png')
						return 'failed - modal buttons not found'
			else:
				print('Connection modal did not appear as expected after clicking Connect.')
				# --- Use Helper for HTML Print ---
				await print_debug_html(page, 'Modal Not Found', 10000)
				# --- End ---
				# Maybe the connection was sent immediately? Unlikely for non-open profiles.
				# Or maybe the connect button click failed silently.
				await page.screenshot(path=f'screenshots/modal_not_found_{time.time()}.png')
				return 'failed - modal not found'

		except Exception as e:
			# This includes TimeoutError waiting for modal
			print(f"Error finding/clicking 'Add a note' or waiting for modal: {e}")
			# Check if modal actually exists despite timeout/error
			modal_check = await page.query_selector('div[role="dialog"][aria-labelledby*="send-invite"]')
			if modal_check:
				print("Modal seems to exist, but finding 'Add Note' failed. Attempting direct send.")
				send_button_direct = await modal_check.query_selector('button[aria-label="Send now"], button:has-text("Send")')
				if send_button_direct and await send_button_direct.is_visible() and not await send_button_direct.is_disabled():
					await send_button_direct.click()
					print('Sent connection request without a note (modal check fallback).')
					return 'success - no note'
				else:
					print('Direct send fallback failed (button not found, invisible, or disabled).')
					# --- Added HTML printing ---
					try:
						modal_html_content = await modal_check.inner_html()
						print('\\n--- Relevant HTML (Modal Fallback Failed) ---')
						print(modal_html_content[:3000])  # Increased snippet size
						print('--- End Relevant HTML ---\\n')
					except Exception as html_e:
						print(f'Could not get modal HTML content for debugging: {html_e}')
					# --- End Added HTML printing ---
					await page.screenshot(path=f'screenshots/modal_fallback_fail_{time.time()}.png')
					return 'failed - add note error / direct send fallback failed'
			else:
				print('Modal not found after error.')
				# --- Use Helper for HTML Print ---
				await print_debug_html(page, 'Modal Not Found After Error', 10000)
				# --- End ---
				await page.screenshot(path=f'screenshots/modal_error_no_modal_{time.time()}.png')
				return 'failed - add note error / modal not found'

		# --- Step 3: Fill the Message ---
		try:
			# Ensure modal context is still valid if possible
			if not modal:  # Should exist if we got here
				modal = await page.query_selector('div[role="dialog"][aria-labelledby*="send-invite"]')
				if not modal:
					print('Modal context lost before filling message.')
					return 'failed - modal disappeared'

			# Use the modal object to query the textarea within it
			message_textarea_selector = 'textarea#custom-message, textarea[name="message"]'
			# Wait for the textarea *within the modal*
			message_textarea = await modal.wait_for_selector(message_textarea_selector, timeout=5000)

			# message_textarea = await modal.query_selector(message_textarea_selector) # Use query_selector after wait
			if not message_textarea:  # Should not happen if wait_for_selector succeeded
				print('Could not find message text area within modal.')
				# --- Added HTML printing ---
				try:
					modal_html_content = await modal.inner_html()
					print('\\n--- Relevant HTML (Text Area Not Found) ---')
					print(modal_html_content[:3000])  # Increased snippet size
					print('--- End Relevant HTML ---\\n')
				except Exception as html_e:
					print(f'Could not get modal HTML content for debugging: {html_e}')
				# --- End Added HTML printing ---
				await page.screenshot(path=f'screenshots/no_textarea_{time.time()}.png')
				return 'failed - message text area not found'

			await message_textarea.fill(message)
			print('Filled message.')
			await asyncio.sleep(1)  # Short delay after filling
		except Exception as e:
			print(f'Error filling message: {e}')
			# --- Added HTML printing ---
			try:
				modal_html_content = await modal.inner_html()
				print('\\n--- Relevant HTML (Fill Message Error) ---')
				print(modal_html_content[:3000])  # Increased snippet size
				print('--- End Relevant HTML ---\\n')
			except Exception as html_e:
				print(f'Could not get modal HTML content for debugging: {html_e}')
			# --- End Added HTML printing ---
			await page.screenshot(path=f'screenshots/fill_message_fail_{time.time()}.png')
			return 'failed - message fill error'

		# --- Step 4: Send the Invitation ---
		try:
			# Re-query modal and send button for robustness
			modal = await page.query_selector('div[role="dialog"][aria-labelledby*="send-invite"]')
			if not modal:
				print('Modal context lost before sending invitation.')
				return 'failed - modal disappeared before send'

			# Look for send button within the modal
			send_button_selector = (
				'button[aria-label="Send invitation"], button[aria-label="Send now"], button:has-text("Send")'  # Broader selector
			)
			send_button = await modal.query_selector(send_button_selector)

			if not send_button or not await send_button.is_visible():
				print('Could not find Send button in modal.')
				# --- Added HTML printing ---
				try:
					modal_html_content = await modal.inner_html()
					print('\\n--- Relevant HTML (Send Button Not Found) ---')
					print(modal_html_content[:3000])  # Increased snippet size
					print('--- End Relevant HTML ---\\n')
				except Exception as html_e:
					print(f'Could not get modal HTML content for debugging: {html_e}')
				# --- End Added HTML printing ---
				await page.screenshot(path=f'screenshots/no_send_button_{time.time()}.png')
				# Check for weekly limit message specifically within the modal or common areas
				limit_reached_modal = await modal.query_selector('text="You\'ve reached the weekly invitation limit"')
				limit_reached_page = await page.locator('text="You\'ve reached the weekly invitation limit"').is_visible()
				if limit_reached_modal or limit_reached_page:
					print('Weekly invitation limit reached.')
					return 'failed - weekly limit reached'
				return 'failed - send button not found'

			# Check if send button is disabled
			if await send_button.is_disabled():
				print('Send button is disabled. Cannot send.')
				# Check for limit message again, as it might disable the button
				limit_reached_modal = await modal.query_selector('text="You\'ve reached the weekly invitation limit"')
				limit_reached_page = await page.locator('text="You\'ve reached the weekly invitation limit"').is_visible()
				if limit_reached_modal or limit_reached_page:
					print('Weekly invitation limit reached (send button disabled).')
					return 'failed - weekly limit reached'
				# --- Added HTML printing ---
				try:
					modal_html_content = await modal.inner_html()
					print('\\n--- Relevant HTML (Send Button Disabled) ---')
					print(modal_html_content[:3000])  # Increased snippet size
					print('--- End Relevant HTML ---\\n')
				except Exception as html_e:
					print(f'Could not get modal HTML content for debugging: {html_e}')
				# --- End Added HTML printing ---
				await page.screenshot(path=f'screenshots/send_disabled_{time.time()}.png')
				return 'failed - send button disabled'

			await send_button.click()
			print('Clicked Send button.')
			await asyncio.sleep(PAGE_LOAD_DELAY)  # Wait for confirmation or next page state

			# Confirmation check (optional but good)
			# Check if modal is gone or success message appears
			modal_check_after_send = await page.query_selector('div[role="dialog"][aria-labelledby*="send-invite"]')
			if not modal_check_after_send:
				print('Modal closed after sending - likely success.')
				status = 'success'
			else:
				# Check for error messages within the modal if it persists
				error_msg = await modal_check_after_send.query_selector('[role="alert"], .artdeco-inline-feedback--error')
				if error_msg:
					error_text = await error_msg.inner_text()
					print(f'Error message shown in modal after send: {error_text}')
					if 'weekly invitation limit' in error_text.lower():
						return 'failed - weekly limit reached'
					status = f'failed - error after send: {error_text[:100]}'  # Truncate long errors
				else:
					print('Modal persisted after sending, unknown state. Assuming success for now.')
					status = 'success - modal persisted'  # Mark potential issue

		except Exception as e:
			print(f'Error clicking Send button: {e}')
			# Check for limit reached after error
			try:
				limit_reached_text = await page.locator('text="reached the weekly invitation limit"').is_visible()
				if limit_reached_text:
					print('Weekly invitation limit reached (on error).')
					return 'failed - weekly limit reached'
			except Exception:
				pass  # Ignore error checking for limit text
			status = 'failed - send error'
			# --- Added HTML printing ---
			try:
				modal_html_content = 'Modal not found or inaccessible.'
				if modal:
					modal_html_content = await modal.inner_html()
				print('\\n--- Relevant HTML (Send Click Error) ---')
				print(modal_html_content[:3000])  # Increased snippet size
				print('--- End Relevant HTML ---\\n')
			except Exception as html_e:
				print(f'Could not get modal HTML content for debugging: {html_e}')
			# --- End Added HTML printing ---
			await page.screenshot(path=f'screenshots/send_click_fail_{time.time()}.png')

	except Exception as e:
		print(f'An unexpected error occurred for {profile_url}: {e}')
		status = f'failed - unexpected error: {e}'
		# Attempt to capture screenshot for debugging
		screenshot_dir = 'screenshots'
		os.makedirs(screenshot_dir, exist_ok=True)
		screenshot_path = os.path.join(screenshot_dir, f'error_screenshot_{time.time()}.png')
		try:
			# --- Use Helper for HTML Print ---
			await print_debug_html(page, 'Unexpected Error', 10000)
			# --- End ---
			await page.screenshot(path=screenshot_path)
			print(f'Error screenshot saved to {screenshot_path}')
		except Exception as screen_e:
			print(f'Could not save screenshot: {screen_e}')

	return status


# --- Main Execution ---


async def main(args):
	# --- Draft Mode ---
	if args.mode == 'draft':
		print('--- Running in Draft Mode ---')

		# --- LLM Initialization (Conditional) ---
		llm = None
		if args.llm_provider == 'deepseek':
			if not deepseek_api_key:
				raise ValueError('DEEPSEEK_API_KEY is not set in the environment variables. Needed for DeepSeek provider.')
			llm = ChatDeepSeek(
				base_url='https://api.deepseek.com/v1',
				model='deepseek-chat',  # Or choose another appropriate model
				api_key=SecretStr(deepseek_api_key),
				temperature=0.1,  # Adjust creativity
			)
			print('Using DeepSeek LLM.')
		elif args.llm_provider == 'openai':
			if not openai_api_key:
				raise ValueError('OPENAI_API_KEY is not set in the environment variables. Needed for OpenAI provider.')
			llm = ChatOpenAI(
				model='gpt-4o',  # Specify GPT-4o
				api_key=SecretStr(openai_api_key),
				temperature=0.1,
			)
			print('Using OpenAI (GPT-4o) LLM.')
		elif args.llm_provider == 'anthropic':
			if not anthropic_api_key:
				raise ValueError('ANTHROPIC_API_KEY is not set in the environment variables. Needed for Anthropic provider.')
			llm = ChatAnthropic(
				model='claude-3-sonnet-20240229',  # Updated model
				api_key=SecretStr(anthropic_api_key),
				temperature=0.1,
			)
			print('Using Anthropic (Claude 3 Sonnet) LLM.')
		else:
			# This case should not be reachable due to argparse choices
			raise ValueError(f'Invalid LLM provider specified: {args.llm_provider}')

		# Load input profiles
		try:
			input_profiles_df = load_input_profiles(args.input_profiles_csv)
			print(f'Loaded {len(input_profiles_df)} profiles from {args.input_profiles_csv}')
		except (FileNotFoundError, ValueError) as e:
			print(f'Error loading input profiles: {e}')
			return

		# Filter relevant profiles
		relevant_input_profiles = input_profiles_df[input_profiles_df['relevancy'] == True].copy()
		if relevant_input_profiles.empty:
			print('No relevant profiles found in the input CSV.')
			return
		print(f'Found {len(relevant_input_profiles)} relevant profiles for message drafting.')

		new_messages_data = []
		# Process relevant profiles
		for index, profile in relevant_input_profiles.iterrows():
			profile_url = profile['linkedin_url']
			if pd.isna(profile_url):
				print(f'Skipping profile (Index {index}, Name: {profile.get("name", "N/A")}) due to missing URL.')
				continue
			profile_school_name = profile.get('school_name')  # Get school name from profile

			print(
				f'\nDrafting message for: {profile.get("name", "N/A")} - {profile.get("position", "N/A")} at {profile.get("company", "N/A")} ({profile_url})'
			)
			generated_msg = await generate_message_with_llm(profile, llm, profile_school_name)  # Pass profile's school name

			if generated_msg:
				print(f'Generated Message: {generated_msg}')
				# Create ConnectionMessage data
				message_entry = ConnectionMessage(
					name=profile.get('name', '[Name N/A]'),
					position=profile.get('position', '[Position N/A]'),
					company=profile.get('company', '[Company N/A]'),
					linkedin_url=profile_url,
					relevancy=profile.get('relevancy', False),
					school_name=profile.get('school_name'),
					company_category=profile.get('company_category'),
					location=profile.get('location'),
					generated_message=generated_msg,
					connection_status='pending',  # Initial status
				)
				new_messages_data.append(message_entry.model_dump())
			else:
				print(f'Failed to generate message for {profile.get("name", "N/A")}. Skipping.')
				# Optionally add to CSV with status 'failed - message generation'
				message_entry = ConnectionMessage(
					name=profile.get('name', '[Name N/A]'),
					position=profile.get('position', '[Position N/A]'),
					company=profile.get('company', '[Company N/A]'),
					linkedin_url=profile_url,
					relevancy=profile.get('relevancy', False),
					school_name=profile.get('school_name'),
					company_category=profile.get('company_category'),
					location=profile.get('location'),
					generated_message=None,
					connection_status='failed - message generation',
				)
				new_messages_data.append(message_entry.model_dump())

			# Optional small delay between LLM calls
			await asyncio.sleep(random.uniform(0.5, 1.5))

		# Combine existing and new messages
		if new_messages_data:
			new_messages_df = pd.DataFrame(new_messages_data)
			# Drop duplicates based on URL, keeping the first (existing) entry
			new_messages_df = new_messages_df.drop_duplicates(subset=['linkedin_url'], keep='first')
			save_connection_messages(new_messages_df, args.messages_csv)
		print('Draft mode finished.')

	# --- Connect Mode ---
	elif args.mode == 'connect':
		print('--- Running in Connect Mode ---')
		# Load connection messages
		messages_df = load_connection_messages(args.messages_csv)
		if messages_df.empty:
			print(f'No connection messages found in {args.messages_csv}. Nothing to connect.')
			return

		# Filter profiles needing connection attempt (pending or retryable failures)
		# Ensure 'generated_message' is not NaN/None for profiles we attempt to connect
		connectable_profiles = messages_df[
			messages_df['connection_status'].isin(RETRYABLE_STATUSES)
			& messages_df['generated_message'].notna()
			& (messages_df['generated_message'] != '')  # Ensure message exists
		].copy()  # Use .copy()

		if connectable_profiles.empty:
			print('No profiles found needing connection requests (check status and generated_message).')
			return

		print(f'Found {len(connectable_profiles)} profiles to attempt connection.')

		# --- Shuffle the order ---
		# connectable_profiles = connectable_profiles.sample(frac=1).reset_index(drop=True)
		profile_indices = connectable_profiles.index.tolist()
		random.shuffle(profile_indices)
		print('Shuffled the order of profiles to connect.')
		# --- End shuffle ---

		# Initialize Browser
		browser = Browser(
			config=BrowserConfig(
				browser_class='chromium',
				browser_binary_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',  # macOS path
			)
		)
		browser_context = None  # Initialize outside try block

		try:
			# Use existing Chrome profile for login session
			browser_context = await browser.new_context(
				config=BrowserContextConfig(
					user_data_dir='~/Library/Application Support/Google/Chrome/Default',  # macOS path
					viewport={'width': 1920, 'height': 1080},
				)
			)
			page = await browser_context.get_current_page()
			await page.set_viewport_size({'width': 1920, 'height': 1080})
			print('Browser context created using existing Chrome profile.')

			# Check if logged into LinkedIn
			await page.goto('https://www.linkedin.com/feed/', wait_until='domcontentloaded')
			await asyncio.sleep(PAGE_LOAD_DELAY)
			# More robust login check
			login_selectors = ['#username', '#password', 'form.login__form', '[data-id="sign-in-form__submit-btn"]']
			is_login_page = False
			for selector in login_selectors:
				if await page.query_selector(selector):
					is_login_page = True
					break

			if is_login_page or 'login' in page.url or 'signup' in page.url:
				print('Error: Not logged into LinkedIn. Please log in manually in your Chrome profile and restart the script.')
				return  # Exit if not logged in

			print('Successfully verified LinkedIn login.')

			# --- Iterate and send requests ---
			# Iterate using index for easy update back to the original DataFrame
			processed_count = 0
			# Iterate through the SHUFFLED indices
			for idx in profile_indices:
				profile = messages_df.loc[idx]  # Get row from original df using shuffled index
				processed_count += 1
				print(
					f'\n--- Processing profile {processed_count}/{len(profile_indices)}: {profile["name"]} ({profile["linkedin_url"]}) ---'
				)

				message = profile['generated_message']
				if pd.isna(message) or not message:
					print('Skipping profile - generated message is missing.')
					messages_df.loc[idx, 'connection_status'] = 'skipped - no message'  # Update using index
					continue

				print(f'Using Message: {message}')
				status = await send_connection_request(browser_context, profile['linkedin_url'], message)
				print(f'Connection Status: {status}')
				messages_df.loc[idx, 'connection_status'] = status  # Update status in main DataFrame using index

				# Save progress periodically (e.g., every 5 attempts)
				if processed_count % 5 == 0:
					save_connection_messages(messages_df, args.messages_csv)

				# Check for weekly limit reached and stop if necessary
				if status == 'failed - weekly limit reached':
					print('Stopping script due to weekly invitation limit.')
					break  # Exit the loop

				# Random delay
				delay = random.uniform(MIN_DELAY_BETWEEN_REQUESTS, MAX_DELAY_BETWEEN_REQUESTS)
				print(f'Waiting for {delay:.2f} seconds before next request...')
				await asyncio.sleep(delay)

		except Exception as e:
			print(f'An error occurred during the connect process: {e}')
		finally:
			# Ensure the browser is closed
			if browser_context:
				await browser_context.close()  # Close context first
			await browser.close()
			print('Browser closed.')
			# Final save
			save_connection_messages(messages_df, args.messages_csv)
			print('Connect mode finished.')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Automate LinkedIn connection requests.')
	parser.add_argument(
		'--mode',
		choices=['draft', 'connect'],
		default='connect',
		help='Operation mode: "draft" to generate messages, "connect" to send requests.',
	)
	parser.add_argument(
		'--input-profiles-csv',
		default=INPUT_PROFILES_CSV_DEFAULT,
		help='Path to the input CSV file with scraped profiles (used in draft mode).',
	)
	parser.add_argument(
		'--messages-csv',
		default=MESSAGES_CSV_DEFAULT,
		help='Path to the CSV file for storing/reading connection messages and statuses.',
	)
	parser.add_argument(
		'--llm-provider',
		choices=['deepseek', 'openai', 'anthropic'],
		default='anthropic',  # Default to anthropic
		help='Choose the LLM provider for message drafting: deepseek, openai, or anthropic (requires respective API key).',
	)

	args = parser.parse_args()

	# Create screenshots directory if it doesn't exist
	os.makedirs('screenshots', exist_ok=True)

	print(f"Running in '{args.mode}' mode.")
	print(f'Using messages file: {args.messages_csv}')
	if args.mode == 'draft':
		print(f'Reading profiles from: {args.input_profiles_csv}')
		print(f'Using LLM Provider: {args.llm_provider}')
	if args.mode == 'connect':
		print('Ensure you are logged into LinkedIn in your default Chrome profile.')

	asyncio.run(main(args))
	print('Script finished.')
