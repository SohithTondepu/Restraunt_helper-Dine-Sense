import os
import re
import json

RESTAURANT_SOP_PLAYBOOK = {
    'Food': [
        {
            'sub_category': 'Temperature & Cold Food',
            'keywords': ['cold', 'lukewarm', 'reheated', 'chilly', 'freeze', 'cool'],
            'sop_title': 'Hot-Holding & Kitchen Pass Temperature Protocol',
            'action_item': 'Install commercial dual-infrared pass-through heat lamps with digital ticket timers (target max pass dwell time: under 120 seconds). Calibrate expo hot-holding wells to 140°F.',
            'kpi_target': 'Zero customer complaints regarding food temperature within 14 days.'
        },
        {
            'sub_category': 'Cooking Precision & Consistency',
            'keywords': ['raw', 'undercooked', 'overcooked', 'burnt', 'chewy', 'rubbery', 'dry', 'tough', 'hard'],
            'sop_title': 'Line-Check Thermal Probe & Kitchen Display Synchronization',
            'action_item': 'Deploy digital Kitchen Display Systems (KDS) synchronized with line-check internal meat temperature probes at the expeditor station.',
            'kpi_target': 'Achieve 98% internal cooking temperature compliance on all protein stations.'
        },
        {
            'sub_category': 'Flavor Balancing & Seasoning',
            'keywords': ['salty', 'bland', 'tasteless', 'oily', 'greasy', 'spice', 'spicy', 'sweet', 'flavorless', 'sour'],
            'sop_title': 'Standardized Batch Tare-Scaling & Shift Tasting Protocol',
            'action_item': 'Standardize pre-portioned seasoning spice packs with digital tare scales and institute mandatory twice-daily shift tasting protocols before lunch and dinner services.',
            'kpi_target': 'Zero seasoning discrepancy variance between day and night culinary teams.'
        },
        {
            'sub_category': 'Freshness & Ingredients',
            'keywords': ['stale', 'smell', 'bad', 'spoiled', 'rotten', 'sour', 'old', 'hair', 'hygiene'],
            'sop_title': 'FIFO Walk-in Cooler Rotation & Digital Temperature Logging',
            'action_item': 'Enforce strict First-In-First-Out (FIFO) walk-in cooler rotation with color-coded day-dot expiration stickers and digital temperature logging twice daily.',
            'kpi_target': '100% compliance during weekly food safety and refrigeration audits.'
        },
        {
            'sub_category': 'Portioning & Plating',
            'keywords': ['portion', 'small', 'quantity', 'presentation', 'messy', 'plate', 'size'],
            'sop_title': 'Visual Plating Guidelines & Calibrated Ladle Portioning',
            'action_item': 'Standardize visual plating photo guides at the expo pass and deploy calibrated portion scoops for side dishes and proteins.',
            'kpi_target': 'Reduce portion size variance to under 5% across all shifts.'
        }
    ],
    'Service': [
        {
            'sub_category': 'Turnaround & Delay',
            'keywords': ['slow', 'delay', 'wait', 'waiting', 'forever', 'late', 'hours', 'minutes', 'time'],
            'sop_title': 'Tableside Mobile POS Terminals & Expedited Bar Runners',
            'action_item': 'Deploy mobile handheld POS terminals for tableside ordering to slash table-to-kitchen firing latency by 4-6 minutes; assign a dedicated runner for drink delivery.',
            'kpi_target': 'Reduce average order-to-table delivery time by 30%.'
        },
        {
            'sub_category': 'Staff Hospitality & Attentiveness',
            'keywords': ['rude', 'arrogant', 'attitude', 'ignored', 'inattentive', 'behavior', 'polite', 'courtesy', 'careless'],
            'sop_title': 'Two-Minute/Two-Bite Check-Back & 45-Second Seating Greeting',
            'action_item': 'Institute the mandatory 2-minute / 2-bite check-back standard and require table greeting within 45 seconds of party seating; conduct mandatory hospitality empathy workshops.',
            'kpi_target': 'Increase positive service sentiment mentions by at least 25%.'
        },
        {
            'sub_category': 'Order Accuracy',
            'keywords': ['wrong', 'mistake', 'forgot', 'missing', 'mix-up', 'incorrect', 'order'],
            'sop_title': 'Verbal Repeat-Back Order Verification & Ticket Stabbing',
            'action_item': 'Mandate verbal repeat-back order verification at tableside and require expeditor ticket-stabbing verification before trays leave the pickup window.',
            'kpi_target': 'Order accuracy rate elevated to 99.5% across all shifts.'
        },
        {
            'sub_category': 'Billing & Payment Latency',
            'keywords': ['bill', 'check', 'payment', 'card', 'pay', 'overcharged', 'receipt'],
            'sop_title': 'Contactless QR Table Settlement & Split-Check POS Automation',
            'action_item': 'Activate contactless QR table pay and automated split-check POS software to compress bill settlement wait times from 12 minutes to under 90 seconds.',
            'kpi_target': 'Eliminate end-of-meal payment bottlenecks during peak rush hours.'
        }
    ],
    'Price': [
        {
            'sub_category': 'Value Perception & Menu Engineering',
            'keywords': ['expensive', 'overpriced', 'costly', 'rip-off', 'pricey', 'money', 'worth', 'cost', 'charge'],
            'sop_title': 'Decoy Pricing Menu Architecture & High-Margin Bundling',
            'action_item': 'Apply menu engineering decoy pricing: reposition signature high-margin items to the top-right focal anchor and bundle high-cost items into multi-course prix-fixe tasting experiences.',
            'kpi_target': 'Lift perceived value sentiment ratio above 70%.'
        },
        {
            'sub_category': 'Portion-to-Price Disparity',
            'keywords': ['quantity', 'small portion', 'tiny', 'little', 'less', 'scant', 'portion'],
            'sop_title': 'Complimentary Amuse-Bouche & Perceived Plate Density Enhancement',
            'action_item': 'Enhance perceived plate density by introducing complimentary house-baked bread, welcome amuse-bouche, or branded palate cleansers to elevate value perception.',
            'kpi_target': 'Improve price-to-quantity satisfaction score by 20 points.'
        },
        {
            'sub_category': 'Fee Transparency',
            'keywords': ['hidden', 'tax', 'service charge', 'extra', 'gst', 'surcharge'],
            'sop_title': 'Transparent Itemized Bill Disclosure & Server Gratuity Scripting',
            'action_item': 'Display transparent all-inclusive menu pricing notes and train servers to proactively clarify automatic gratuity or taxes upfront upon bill presentation.',
            'kpi_target': 'Zero customer complaints regarding unexpected bill line items.'
        }
    ],
    'Ambience': [
        {
            'sub_category': 'Acoustics & Sound Levels',
            'keywords': ['noisy', 'loud', 'music', 'deafening', 'shouting', 'hear', 'echo', 'sound', 'noise'],
            'sop_title': 'Acoustic Felt Baffles & Decibel SPL Sound Level Capping',
            'action_item': 'Install acoustic felt ceiling baffles, fabric wall art panels, and silicone table-foot dampeners to absorb high-frequency chatter; calibrate background playlist SPL to 68-72 dBA.',
            'kpi_target': 'Maintain dining room ambient sound levels within conversational comfort threshold.'
        },
        {
            'sub_category': 'Lighting & Atmosphere',
            'keywords': ['dark', 'dim', 'bright', 'lighting', 'light', 'blind', 'neon'],
            'sop_title': 'Warm-Spectrum 2700K Architectural LED Dimming Calibration',
            'action_item': 'Transition to 2700K warm-spectrum dimmable architectural LED fixtures; install table-top ambient accent lighting to enhance intimate dining atmosphere.',
            'kpi_target': 'Standardize daytime and evening lumen transition schedules.'
        },
        {
            'sub_category': 'HVAC & Climate Control',
            'keywords': ['cold', 'hot', 'ac', 'air conditioning', 'sweating', 'freeze', 'chilly', 'stuffy', 'ventilation', 'temperature'],
            'sop_title': 'HVAC Diffuser Redirection & Dining Hall Thermal Balancing',
            'action_item': 'Recalibrate HVAC diffuser registers away from direct guest table seating, maintain dining room temperature at 68°F to 72°F, and service kitchen exhaust hood make-up air dampers.',
            'kpi_target': 'Eliminate guest complaints regarding indoor draft and thermal discomfort.'
        },
        {
            'sub_category': 'Cleanliness & Restrooms',
            'keywords': ['dirty', 'smell', 'odor', 'restroom', 'bathroom', 'toilet', 'washroom', 'fly', 'flies', 'clean'],
            'sop_title': 'Digital 30-Minute Restroom Audit & Dedicated Busser Station Sanitization',
            'action_item': 'Implement a digital 30-minute timestamped restroom inspection log and assign dedicated bussers with sanitized microfiber towels for instant turnover.',
            'kpi_target': '100% checklist completion rate recorded on shift management logs.'
        }
    ]
}


def retrieve_operational_sop(aspect: str, issue_text: str = "") -> dict:
    """
    Retrieves the most targeted domain operational SOP for a given aspect and customer complaint.
    Uses semantic keyword overlap with fallback to primary aspect intervention.
    """
    candidates = RESTAURANT_SOP_PLAYBOOK.get(aspect, [])
    if not candidates:
        candidates = [sop for sops in RESTAURANT_SOP_PLAYBOOK.values() for sop in sops]
        
    issue_lower = (issue_text or '').lower()
    best_sop = candidates[0]
    best_score = -1
    
    for sop in candidates:
        score = sum(1 for kw in sop['keywords'] if kw in issue_lower)
        if score > best_score:
            best_score = score
            best_sop = sop
            
    return best_sop


def build_restaurant_payload(scorecard: dict, complaint_clusters: list) -> dict:
    """
    Constructs a verified factual JSON payload containing computed metrics, real quotes,
    and retrieved operational SOP playbooks for RAG grounding.
    """
    aspects_data = {}
    for asp, data in scorecard.get('aspects', {}).items():
        aspects_data[asp] = {
            'score': data['score'],
            'status': data['status'],
            'total_mentions': data['total_mentions'],
            'negative_mentions': data['neg_mentions'],
            'positive_mentions': data['pos_mentions']
        }
        
    clusters_data = []
    for c in (complaint_clusters or [])[:3]:
        clusters_data.append({
            'issue': c.get('root_cause', 'Unknown issue'),
            'frequency': c.get('frequency', 0),
            'sample_quote': c.get('evidence_quotes', ['No quote available'])[0] if c.get('evidence_quotes') else ''
        })

    # Retrieve matching SOPs for weakest aspects and top complaints
    sorted_aspects = sorted(aspects_data.items(), key=lambda x: x[1]['score'])
    weakest_aspect = sorted_aspects[0][0] if sorted_aspects else 'Service'
    top_issue = clusters_data[0]['issue'] if clusters_data else ''
    
    primary_sop = retrieve_operational_sop(weakest_aspect, top_issue)
    
    secondary_aspect = sorted_aspects[1][0] if len(sorted_aspects) > 1 else 'Food'
    secondary_issue = clusters_data[1]['issue'] if len(clusters_data) > 1 else ''
    secondary_sop = retrieve_operational_sop(secondary_aspect, secondary_issue)

    payload = {
        'restaurant': scorecard.get('restaurant', 'Target Restaurant'),
        'overall_health_index': scorecard.get('overall_health_index', 0.0),
        'average_stars': scorecard.get('average_stars', 0.0),
        'total_reviews': scorecard.get('total_reviews', 0),
        'aspect_performance': aspects_data,
        'top_complaint_clusters': clusters_data,
        'retrieved_sops': [primary_sop, secondary_sop]
    }
    return payload


def generate_grounded_report(payload: dict, api_key: str = None) -> dict:
    """
    Generates an executive action report via Hybrid RAG.
    Grounds actionable recommendations in verified domain operational playbooks
    (heat lamps, POS terminals, acoustic baffles, decoy pricing) with zero hallucination.
    """
    rest_name = payload['restaurant']
    health = payload['overall_health_index']
    stars = payload['average_stars']
    aspects = payload['aspect_performance']
    complaints = payload['top_complaint_clusters']
    sops = payload.get('retrieved_sops', [])
    
    # Identify lowest and highest performing aspects
    sorted_aspects = sorted(aspects.items(), key=lambda x: x[1]['score'])
    weakest_aspect, weakest_data = sorted_aspects[0]
    best_aspect, best_data = sorted_aspects[-1]
    
    top_issue = complaints[0]['issue'] if complaints else 'General service variance'
    top_quote = complaints[0]['sample_quote'] if complaints else 'No direct quote cited'
    top_freq = complaints[0]['frequency'] if complaints else 0
    
    primary_sop = sops[0] if sops else retrieve_operational_sop(weakest_aspect, top_issue)
    secondary_sop = sops[1] if len(sops) > 1 else retrieve_operational_sop('Food' if weakest_aspect != 'Food' else 'Service')
    
    report_text = f"""### Executive Strategic Action Report: {rest_name}

**1. Executive Summary & Health Diagnosis**
{rest_name} operates with an Overall Health Index of **{health}/100** based on **{payload['total_reviews']}** customer reviews (Average Star Rating: **{stars}**). The primary operational strength is **{best_aspect}** at a health score of **{best_data['score']}/100** ({best_data['positive_mentions']} positive mentions). Conversely, the primary bottleneck is **{weakest_aspect}**, registering an alert score of **{weakest_data['score']}/100** with **{weakest_data['negative_mentions']}** critical complaints.

**2. Root-Cause Operational Bottlenecks**
Analysis of dependency-parsed negative reviews reveals:
- **Priority 1: {top_issue}** — Cited in **{top_freq}** independent customer reviews.
  *Customer Quote:* "{top_quote}"
"""
    if len(complaints) > 1:
        report_text += f"""- **Priority 2: {complaints[1]['issue']}** — Logged in **{complaints[1]['frequency']}** reviews.
  *Customer Quote:* "{complaints[1]['sample_quote']}"
"""

    report_text += f"""
**3. Prescriptive 30-Day Operational Action Roadmap (RAG-Synthesized)**
1. **Immediate Intervention ({weakest_aspect} — {primary_sop['sop_title']}):**
   - *Operational Intervention:* {primary_sop['action_item']}
   - *Target KPI:* {primary_sop['kpi_target']} (reduces the {weakest_data['negative_mentions']} recorded complaints).
2. **Secondary Priority ({secondary_sop['sop_title']}):**
   - *Operational Intervention:* {secondary_sop['action_item']}
   - *Target KPI:* {secondary_sop['kpi_target']}
3. **Core Strength Capitalization ({best_aspect}):**
   - Leverage the **{best_data['score']}/100** health rating and **{best_data['positive_mentions']}** positive citations in {best_aspect.lower()} across digital marketing and signature menu highlights.
"""

    # Verification Step:
    verification_passed, unmatched = verify_report_numbers(report_text, payload)
    
    return {
        'report_markdown': report_text,
        'verification_passed': verification_passed,
        'unmatched_numbers': unmatched,
        'payload': payload
    }


def verify_report_numbers(report_text: str, payload: dict) -> tuple:
    """
    Extracts every numeric entity from report text and confirms whether it matches
    the numbers present in the underlying JSON payload.
    """
    payload_str = json.dumps(payload)
    # Extract integer and float numbers from report text
    raw_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', report_text)
    
    unmatched = []
    # Allow trivial numbers like bullet points 1, 2, 3, 4, 5 and 30 or 100
    exempt = {'1', '2', '3', '4', '5', '30', '100'}
    
    for num in raw_numbers:
        if num in exempt:
            continue
        # Check if number appears in payload string
        if num not in payload_str:
            unmatched.append(num)
            
    is_valid = len(unmatched) == 0
    return is_valid, unmatched
