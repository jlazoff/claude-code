#!/usr/bin/env python3
"""
Digital Twin Orchestrator
Creates and manages a digital twin based on user interactions and preferences
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import hashlib
from pydantic import BaseModel, Field
from datetime import datetime, timedelta

# Local framework integration
from local_agentic_framework import LocalAgenticFramework, KnowledgeGraphNode, DataLakeRecord

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserInteraction(BaseModel):
    """Model for user interactions"""
    interaction_id: str = Field(..., description="Unique interaction ID")
    timestamp: str = Field(..., description="Interaction timestamp")
    interaction_type: str = Field(..., description="Type of interaction")
    context: Dict[str, Any] = Field(default_factory=dict, description="Interaction context")
    user_choice: str = Field(..., description="User's choice or decision")
    available_options: List[Dict[str, Any]] = Field(default_factory=list, description="Options presented")
    response_time: float = Field(default=0.0, description="Time taken to respond")
    confidence_level: float = Field(default=0.5, description="User's confidence in choice")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class UserPreferences(BaseModel):
    """Model for user preferences learned over time"""
    preference_id: str = Field(..., description="Unique preference ID")
    category: str = Field(..., description="Preference category")
    preference_type: str = Field(..., description="Type of preference")
    value: Union[str, int, float, bool, List, Dict] = Field(..., description="Preference value")
    confidence: float = Field(default=0.5, description="Confidence in this preference")
    frequency: int = Field(default=1, description="How often this preference is observed")
    last_observed: str = Field(..., description="Last time this preference was observed")
    context_patterns: List[str] = Field(default_factory=list, description="Context patterns where this applies")

class DecisionPattern(BaseModel):
    """Model for learned decision patterns"""
    pattern_id: str = Field(..., description="Unique pattern ID")
    pattern_name: str = Field(..., description="Human-readable pattern name")
    trigger_conditions: Dict[str, Any] = Field(default_factory=dict, description="Conditions that trigger this pattern")
    typical_choice: str = Field(..., description="Most common choice for this pattern")
    alternative_choices: List[str] = Field(default_factory=list, description="Alternative choices and their frequencies")
    success_rate: float = Field(default=0.5, description="Success rate of this pattern")
    adaptability_score: float = Field(default=0.5, description="How adaptable this pattern is")
    usage_frequency: int = Field(default=1, description="How often this pattern is used")

class DigitalTwinProfile(BaseModel):
    """Complete digital twin profile"""
    twin_id: str = Field(..., description="Unique digital twin ID")
    created_at: str = Field(..., description="Creation timestamp")
    last_updated: str = Field(..., description="Last update timestamp")
    interaction_count: int = Field(default=0, description="Total interactions")
    
    # Learned characteristics
    decision_speed: float = Field(default=5.0, description="Average decision time in seconds")
    risk_tolerance: float = Field(default=0.5, description="Risk tolerance level (0-1)")
    complexity_preference: str = Field(default="medium", description="Preferred complexity level")
    automation_preference: float = Field(default=0.7, description="Preference for automation (0-1)")
    learning_style: str = Field(default="balanced", description="Preferred learning style")
    
    # Technical preferences
    preferred_languages: List[str] = Field(default_factory=list, description="Preferred programming languages")
    preferred_frameworks: List[str] = Field(default_factory=list, description="Preferred frameworks")
    preferred_tools: List[str] = Field(default_factory=list, description="Preferred development tools")
    working_hours: Dict[str, Any] = Field(default_factory=dict, description="Preferred working hours")
    
    # Behavioral patterns
    decision_patterns: List[str] = Field(default_factory=list, description="Associated decision pattern IDs")
    preferences: List[str] = Field(default_factory=list, description="Associated preference IDs")
    interaction_history: List[str] = Field(default_factory=list, description="Recent interaction IDs")

class DigitalTwinOrchestrator:
    """Orchestrates creation and management of digital twin"""
    
    def __init__(self, framework: LocalAgenticFramework):
        self.framework = framework
        self.foundation_dir = Path("foundation_data")
        self.twin_dir = self.foundation_dir / "digital_twin"
        self.interactions_dir = self.twin_dir / "interactions"
        self.profiles_dir = self.twin_dir / "profiles"
        
        # Create directories
        for dir_path in [self.twin_dir, self.interactions_dir, self.profiles_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Twin state
        self.current_profile: Optional[DigitalTwinProfile] = None
        self.interactions: Dict[str, UserInteraction] = {}
        self.preferences: Dict[str, UserPreferences] = {}
        self.decision_patterns: Dict[str, DecisionPattern] = {}
        
        # Learning parameters
        self.learning_rate = 0.1
        self.pattern_threshold = 3  # Minimum interactions to form a pattern
        self.confidence_decay = 0.95  # How confidence decays over time
        
        # Initialize twin
        asyncio.create_task(self.initialize_digital_twin())
        
        logger.info("Digital Twin Orchestrator initialized")

    async def initialize_digital_twin(self):
        """Initialize or load existing digital twin"""
        logger.info("🧬 Initializing Digital Twin...")
        
        # Try to load existing profile
        existing_profiles = list(self.profiles_dir.glob("profile_*.json"))
        
        if existing_profiles:
            # Load most recent profile
            latest_profile = sorted(existing_profiles, key=lambda x: x.stat().st_mtime)[-1]
            try:
                with open(latest_profile) as f:
                    profile_data = json.load(f)
                    self.current_profile = DigitalTwinProfile(**profile_data)
                logger.info(f"✅ Loaded existing digital twin profile: {self.current_profile.twin_id}")
            except Exception as e:
                logger.warning(f"Failed to load existing profile: {e}")
                await self.create_new_profile()
        else:
            await self.create_new_profile()
        
        # Load interactions, preferences, and patterns
        await self.load_twin_data()

    async def create_new_profile(self):
        """Create a new digital twin profile"""
        logger.info("🆕 Creating new digital twin profile...")
        
        twin_id = f"twin_{int(time.time())}"
        
        self.current_profile = DigitalTwinProfile(
            twin_id=twin_id,
            created_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            working_hours={
                "timezone": "UTC",
                "start_hour": 9,
                "end_hour": 17,
                "days": ["monday", "tuesday", "wednesday", "thursday", "friday"]
            }
        )
        
        # Save initial profile
        await self.save_profile()
        
        logger.info(f"✅ Created new digital twin: {twin_id}")

    async def load_twin_data(self):
        """Load interactions, preferences, and patterns"""
        try:
            # Load interactions
            for interaction_file in self.interactions_dir.glob("interaction_*.json"):
                with open(interaction_file) as f:
                    interaction_data = json.load(f)
                    interaction = UserInteraction(**interaction_data)
                    self.interactions[interaction.interaction_id] = interaction
            
            # Load preferences and patterns from knowledge graph
            await self.load_from_knowledge_graph()
            
            logger.info(f"✅ Loaded {len(self.interactions)} interactions, {len(self.preferences)} preferences, {len(self.decision_patterns)} patterns")
            
        except Exception as e:
            logger.warning(f"Failed to load twin data: {e}")

    async def load_from_knowledge_graph(self):
        """Load preferences and patterns from knowledge graph"""
        # This would query the knowledge graph for digital twin data
        # For now, we'll implement basic functionality
        pass

    async def record_interaction(self, interaction_type: str, context: Dict[str, Any], 
                               user_choice: str, available_options: List[Dict[str, Any]], 
                               response_time: float = 0.0, confidence: float = 0.5) -> UserInteraction:
        """Record a user interaction for learning"""
        
        interaction_id = f"int_{int(time.time() * 1000)}"
        
        interaction = UserInteraction(
            interaction_id=interaction_id,
            timestamp=datetime.now().isoformat(),
            interaction_type=interaction_type,
            context=context,
            user_choice=user_choice,
            available_options=available_options,
            response_time=response_time,
            confidence_level=confidence,
            metadata={
                "twin_id": self.current_profile.twin_id if self.current_profile else "unknown"
            }
        )
        
        # Store interaction
        self.interactions[interaction_id] = interaction
        
        # Save to file
        interaction_file = self.interactions_dir / f"interaction_{interaction_id}.json"
        with open(interaction_file, 'w') as f:
            json.dump(interaction.model_dump(), f, indent=2, default=str)
        
        # Store in knowledge graph
        await self.store_interaction_in_graph(interaction)
        
        # Update profile
        if self.current_profile:
            self.current_profile.interaction_count += 1
            self.current_profile.last_updated = datetime.now().isoformat()
            
            # Update interaction history (keep last 100)
            self.current_profile.interaction_history.append(interaction_id)
            if len(self.current_profile.interaction_history) > 100:
                self.current_profile.interaction_history = self.current_profile.interaction_history[-100:]
        
        # Learn from interaction
        await self.learn_from_interaction(interaction)
        
        logger.info(f"📝 Recorded interaction: {interaction_type} -> {user_choice}")
        
        return interaction

    async def store_interaction_in_graph(self, interaction: UserInteraction):
        """Store interaction in knowledge graph"""
        node = KnowledgeGraphNode(
            node_id=f"interaction_{interaction.interaction_id}",
            node_type="user_interaction",
            content=interaction.model_dump(),
            metadata={
                "twin_id": self.current_profile.twin_id if self.current_profile else "unknown",
                "interaction_type": interaction.interaction_type
            },
            created_at=interaction.timestamp,
            updated_at=interaction.timestamp
        )
        
        await self.framework.store_knowledge_node(node)

    async def learn_from_interaction(self, interaction: UserInteraction):
        """Learn preferences and patterns from interaction"""
        
        # Learn preferences
        await self.update_preferences(interaction)
        
        # Learn decision patterns
        await self.update_decision_patterns(interaction)
        
        # Update profile characteristics
        await self.update_profile_characteristics(interaction)

    async def update_preferences(self, interaction: UserInteraction):
        """Update user preferences based on interaction"""
        
        # Extract preferences from interaction
        preferences_to_update = []
        
        # Language preferences
        if "language" in interaction.context:
            preferences_to_update.append({
                "category": "technical",
                "preference_type": "programming_language",
                "value": interaction.context["language"],
                "context_pattern": interaction.interaction_type
            })
        
        # Framework preferences
        if "framework" in interaction.context:
            preferences_to_update.append({
                "category": "technical", 
                "preference_type": "framework",
                "value": interaction.context["framework"],
                "context_pattern": interaction.interaction_type
            })
        
        # Decision speed preference
        if interaction.response_time > 0:
            preferences_to_update.append({
                "category": "behavioral",
                "preference_type": "decision_speed",
                "value": interaction.response_time,
                "context_pattern": interaction.interaction_type
            })
        
        # Complexity preference
        if "complexity" in interaction.context:
            preferences_to_update.append({
                "category": "behavioral",
                "preference_type": "complexity_preference", 
                "value": interaction.context["complexity"],
                "context_pattern": interaction.interaction_type
            })
        
        # Update or create preferences
        for pref_data in preferences_to_update:
            await self.update_preference(pref_data, interaction)

    async def update_preference(self, pref_data: Dict[str, Any], interaction: UserInteraction):
        """Update or create a specific preference"""
        
        # Create preference key
        pref_key = f"{pref_data['category']}_{pref_data['preference_type']}_{str(pref_data['value'])}"
        
        if pref_key in self.preferences:
            # Update existing preference
            pref = self.preferences[pref_key]
            pref.frequency += 1
            pref.last_observed = interaction.timestamp
            pref.confidence = min(1.0, pref.confidence + self.learning_rate)
            
            # Add context pattern if new
            if pref_data["context_pattern"] not in pref.context_patterns:
                pref.context_patterns.append(pref_data["context_pattern"])
                
        else:
            # Create new preference
            preference_id = f"pref_{hashlib.md5(pref_key.encode()).hexdigest()[:8]}"
            
            pref = UserPreferences(
                preference_id=preference_id,
                category=pref_data["category"],
                preference_type=pref_data["preference_type"],
                value=pref_data["value"],
                confidence=self.learning_rate,
                frequency=1,
                last_observed=interaction.timestamp,
                context_patterns=[pref_data["context_pattern"]]
            )
            
            self.preferences[pref_key] = pref
        
        # Update profile if relevant
        if self.current_profile:
            if pref_data["preference_type"] == "programming_language":
                if pref_data["value"] not in self.current_profile.preferred_languages:
                    self.current_profile.preferred_languages.append(str(pref_data["value"]))
            elif pref_data["preference_type"] == "framework":
                if pref_data["value"] not in self.current_profile.preferred_frameworks:
                    self.current_profile.preferred_frameworks.append(str(pref_data["value"]))

    async def update_decision_patterns(self, interaction: UserInteraction):
        """Update decision patterns based on interaction"""
        
        # Create pattern signature based on context
        context_signature = self.create_context_signature(interaction.context, interaction.interaction_type)
        
        pattern_key = f"pattern_{hashlib.md5(context_signature.encode()).hexdigest()[:8]}"
        
        if pattern_key in self.decision_patterns:
            # Update existing pattern
            pattern = self.decision_patterns[pattern_key]
            pattern.usage_frequency += 1
            
            # Update typical choice if this choice is more frequent
            if interaction.user_choice == pattern.typical_choice:
                pattern.success_rate = min(1.0, pattern.success_rate + self.learning_rate)
            else:
                # Add to alternatives or update frequency
                choice_found = False
                for alt_choice in pattern.alternative_choices:
                    if alt_choice == interaction.user_choice:
                        choice_found = True
                        break
                
                if not choice_found:
                    pattern.alternative_choices.append(interaction.user_choice)
                
                # Potentially update typical choice if this one becomes more frequent
                # This would require more sophisticated tracking
                
        else:
            # Create new pattern if we have enough similar interactions
            similar_interactions = self.find_similar_interactions(interaction)
            
            if len(similar_interactions) >= self.pattern_threshold:
                pattern_id = f"pattern_{int(time.time())}"
                
                pattern = DecisionPattern(
                    pattern_id=pattern_id,
                    pattern_name=f"{interaction.interaction_type}_{len(self.decision_patterns) + 1}",
                    trigger_conditions=interaction.context,
                    typical_choice=interaction.user_choice,
                    alternative_choices=[],
                    success_rate=0.5,
                    adaptability_score=0.5,
                    usage_frequency=1
                )
                
                self.decision_patterns[pattern_key] = pattern
                
                # Add to profile
                if self.current_profile and pattern_id not in self.current_profile.decision_patterns:
                    self.current_profile.decision_patterns.append(pattern_id)

    def create_context_signature(self, context: Dict[str, Any], interaction_type: str) -> str:
        """Create a signature for the interaction context"""
        # Sort context items for consistent signature
        sorted_context = sorted(context.items())
        context_str = f"{interaction_type}:" + ":".join(f"{k}={v}" for k, v in sorted_context)
        return context_str

    def find_similar_interactions(self, interaction: UserInteraction) -> List[UserInteraction]:
        """Find interactions similar to the given one"""
        similar = []
        
        for other_interaction in self.interactions.values():
            if other_interaction.interaction_type == interaction.interaction_type:
                # Check context similarity
                similarity_score = self.calculate_context_similarity(
                    interaction.context, 
                    other_interaction.context
                )
                
                if similarity_score > 0.7:  # 70% similarity threshold
                    similar.append(other_interaction)
        
        return similar

    def calculate_context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Calculate similarity between two contexts"""
        if not context1 or not context2:
            return 0.0
        
        # Get all keys
        all_keys = set(context1.keys()) | set(context2.keys())
        if not all_keys:
            return 1.0
        
        matching_keys = 0
        for key in all_keys:
            if key in context1 and key in context2:
                if context1[key] == context2[key]:
                    matching_keys += 1
        
        return matching_keys / len(all_keys)

    async def update_profile_characteristics(self, interaction: UserInteraction):
        """Update profile characteristics based on interaction"""
        if not self.current_profile:
            return
        
        # Update decision speed
        if interaction.response_time > 0:
            current_speed = self.current_profile.decision_speed
            new_speed = current_speed * (1 - self.learning_rate) + interaction.response_time * self.learning_rate
            self.current_profile.decision_speed = new_speed
        
        # Update risk tolerance based on choices
        if "risk_level" in interaction.context:
            risk_choice = interaction.context["risk_level"]
            if risk_choice in ["high", "medium", "low"]:
                risk_value = {"low": 0.2, "medium": 0.5, "high": 0.8}[risk_choice]
                current_risk = self.current_profile.risk_tolerance
                new_risk = current_risk * (1 - self.learning_rate) + risk_value * self.learning_rate
                self.current_profile.risk_tolerance = new_risk
        
        # Update automation preference
        if "automation" in interaction.user_choice.lower():
            current_auto = self.current_profile.automation_preference
            new_auto = current_auto * (1 - self.learning_rate) + 0.8 * self.learning_rate
            self.current_profile.automation_preference = new_auto
        elif "manual" in interaction.user_choice.lower():
            current_auto = self.current_profile.automation_preference
            new_auto = current_auto * (1 - self.learning_rate) + 0.3 * self.learning_rate
            self.current_profile.automation_preference = new_auto

    async def predict_user_choice(self, interaction_type: str, context: Dict[str, Any], 
                                available_options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict user choice based on learned patterns"""
        
        # Find matching decision patterns
        context_signature = self.create_context_signature(context, interaction_type)
        pattern_key = f"pattern_{hashlib.md5(context_signature.encode()).hexdigest()[:8]}"
        
        prediction = {
            "predicted_choice": None,
            "confidence": 0.0,
            "reasoning": [],
            "alternatives": [],
            "learning_opportunity": True
        }
        
        # Check for exact pattern match
        if pattern_key in self.decision_patterns:
            pattern = self.decision_patterns[pattern_key]
            prediction["predicted_choice"] = pattern.typical_choice
            prediction["confidence"] = pattern.success_rate
            prediction["reasoning"].append(f"Found exact pattern match with {pattern.usage_frequency} uses")
            prediction["alternatives"] = pattern.alternative_choices
            prediction["learning_opportunity"] = False
        else:
            # Look for similar patterns
            best_match = None
            best_similarity = 0.0
            
            for pattern in self.decision_patterns.values():
                similarity = self.calculate_context_similarity(context, pattern.trigger_conditions)
                if similarity > best_similarity and similarity > 0.5:
                    best_similarity = similarity
                    best_match = pattern
            
            if best_match:
                prediction["predicted_choice"] = best_match.typical_choice
                prediction["confidence"] = best_similarity * best_match.success_rate
                prediction["reasoning"].append(f"Found similar pattern with {best_similarity:.2f} similarity")
                prediction["alternatives"] = best_match.alternative_choices
            else:
                # Use preferences to make prediction
                preference_score = await self.score_options_by_preferences(available_options, context)
                if preference_score:
                    best_option = max(preference_score.items(), key=lambda x: x[1])
                    prediction["predicted_choice"] = best_option[0]
                    prediction["confidence"] = min(0.7, best_option[1])
                    prediction["reasoning"].append("Based on learned preferences")
        
        return prediction

    async def score_options_by_preferences(self, options: List[Dict[str, Any]], 
                                         context: Dict[str, Any]) -> Dict[str, float]:
        """Score available options based on learned preferences"""
        scores = {}
        
        for option in options:
            option_id = option.get("id", str(option))
            score = 0.0
            
            # Score based on technical preferences
            if "language" in option:
                if option["language"] in self.current_profile.preferred_languages:
                    score += 0.3
            
            if "framework" in option:
                if option["framework"] in self.current_profile.preferred_frameworks:
                    score += 0.3
            
            # Score based on complexity
            if "complexity" in option and self.current_profile:
                option_complexity = option["complexity"]
                if option_complexity == self.current_profile.complexity_preference:
                    score += 0.2
            
            # Score based on automation level
            if "automation_level" in option and self.current_profile:
                auto_level = option["automation_level"]
                auto_score = {"low": 0.2, "medium": 0.5, "high": 0.8}.get(auto_level, 0.5)
                preference_distance = abs(auto_score - self.current_profile.automation_preference)
                score += 0.2 * (1 - preference_distance)
            
            scores[option_id] = score
        
        return scores

    async def save_profile(self):
        """Save current profile to disk"""
        if not self.current_profile:
            return
        
        profile_file = self.profiles_dir / f"profile_{self.current_profile.twin_id}.json"
        with open(profile_file, 'w') as f:
            json.dump(self.current_profile.model_dump(), f, indent=2, default=str)
        
        # Also save preferences and patterns
        await self.save_twin_data()

    async def save_twin_data(self):
        """Save preferences and patterns"""
        
        # Save preferences
        prefs_file = self.twin_dir / "preferences.json"
        with open(prefs_file, 'w') as f:
            json.dump({k: v.model_dump() for k, v in self.preferences.items()}, f, indent=2, default=str)
        
        # Save patterns
        patterns_file = self.twin_dir / "patterns.json"
        with open(patterns_file, 'w') as f:
            json.dump({k: v.model_dump() for k, v in self.decision_patterns.items()}, f, indent=2, default=str)

    async def get_twin_status(self) -> Dict[str, Any]:
        """Get comprehensive digital twin status"""
        if not self.current_profile:
            return {"status": "not_initialized"}
        
        return {
            "status": "active",
            "twin_id": self.current_profile.twin_id,
            "created_at": self.current_profile.created_at,
            "last_updated": self.current_profile.last_updated,
            "statistics": {
                "total_interactions": self.current_profile.interaction_count,
                "total_preferences": len(self.preferences),
                "total_patterns": len(self.decision_patterns),
                "recent_interactions": len(self.current_profile.interaction_history)
            },
            "characteristics": {
                "decision_speed": self.current_profile.decision_speed,
                "risk_tolerance": self.current_profile.risk_tolerance,
                "complexity_preference": self.current_profile.complexity_preference,
                "automation_preference": self.current_profile.automation_preference,
                "learning_style": self.current_profile.learning_style
            },
            "preferences": {
                "languages": self.current_profile.preferred_languages,
                "frameworks": self.current_profile.preferred_frameworks,
                "tools": self.current_profile.preferred_tools
            },
            "learning_metrics": {
                "pattern_accuracy": self.calculate_pattern_accuracy(),
                "preference_consistency": self.calculate_preference_consistency(),
                "adaptation_rate": self.calculate_adaptation_rate()
            }
        }

    def calculate_pattern_accuracy(self) -> float:
        """Calculate accuracy of decision patterns"""
        if not self.decision_patterns:
            return 0.0
        
        total_accuracy = sum(pattern.success_rate for pattern in self.decision_patterns.values())
        return total_accuracy / len(self.decision_patterns)

    def calculate_preference_consistency(self) -> float:
        """Calculate consistency of preferences"""
        if not self.preferences:
            return 0.0
        
        total_confidence = sum(pref.confidence for pref in self.preferences.values())
        return total_confidence / len(self.preferences)

    def calculate_adaptation_rate(self) -> float:
        """Calculate how quickly the twin adapts"""
        if self.current_profile.interaction_count < 10:
            return 0.5  # Not enough data
        
        # Base adaptation on how frequently patterns and preferences are updated
        recent_adaptations = 0
        total_items = len(self.preferences) + len(self.decision_patterns)
        
        # Simple metric based on recent activity
        return min(1.0, recent_adaptations / max(total_items, 1))

async def main():
    """Test the Digital Twin Orchestrator"""
    # Initialize framework
    framework = LocalAgenticFramework()
    await asyncio.sleep(3)  # Wait for framework initialization
    
    # Initialize digital twin
    twin = DigitalTwinOrchestrator(framework)
    await asyncio.sleep(2)  # Wait for twin initialization
    
    # Simulate some interactions
    test_interactions = [
        {
            "interaction_type": "tool_selection",
            "context": {"task": "code_generation", "language": "python", "complexity": "medium"},
            "user_choice": "automated_generation",
            "available_options": [
                {"id": "manual_coding", "complexity": "high"},
                {"id": "automated_generation", "complexity": "medium"},
                {"id": "template_based", "complexity": "low"}
            ],
            "response_time": 3.5,
            "confidence": 0.8
        },
        {
            "interaction_type": "framework_choice",
            "context": {"project_type": "web_api", "language": "python", "complexity": "medium"},
            "user_choice": "fastapi",
            "available_options": [
                {"id": "django", "complexity": "high"},
                {"id": "fastapi", "complexity": "medium"},
                {"id": "flask", "complexity": "low"}
            ],
            "response_time": 2.1,
            "confidence": 0.9
        },
        {
            "interaction_type": "deployment_strategy",
            "context": {"environment": "production", "scale": "medium", "risk_level": "low"},
            "user_choice": "containerized_deployment",
            "available_options": [
                {"id": "manual_deployment", "automation_level": "low"},
                {"id": "containerized_deployment", "automation_level": "high"},
                {"id": "serverless_deployment", "automation_level": "high"}
            ],
            "response_time": 4.2,
            "confidence": 0.7
        }
    ]
    
    # Record interactions
    print("🧬 Recording test interactions...")
    for interaction_data in test_interactions:
        await twin.record_interaction(**interaction_data)
        await asyncio.sleep(1)
    
    # Test prediction
    print("\n🔮 Testing prediction...")
    prediction = await twin.predict_user_choice(
        "tool_selection",
        {"task": "code_generation", "language": "python", "complexity": "medium"},
        [
            {"id": "manual_coding", "complexity": "high"},
            {"id": "automated_generation", "complexity": "medium"},
            {"id": "ai_assisted", "complexity": "medium"}
        ]
    )
    
    print(f"   Predicted Choice: {prediction['predicted_choice']}")
    print(f"   Confidence: {prediction['confidence']:.2f}")
    print(f"   Reasoning: {', '.join(prediction['reasoning'])}")
    
    # Get twin status
    status = await twin.get_twin_status()
    
    print(f"\n🤖 Digital Twin Status:")
    print(f"   Twin ID: {status['twin_id']}")
    print(f"   Interactions: {status['statistics']['total_interactions']}")
    print(f"   Preferences: {status['statistics']['total_preferences']}")
    print(f"   Patterns: {status['statistics']['total_patterns']}")
    print(f"   Decision Speed: {status['characteristics']['decision_speed']:.1f}s")
    print(f"   Risk Tolerance: {status['characteristics']['risk_tolerance']:.2f}")
    print(f"   Automation Preference: {status['characteristics']['automation_preference']:.2f}")
    print(f"   Preferred Languages: {', '.join(status['preferences']['languages'])}")
    print(f"   Pattern Accuracy: {status['learning_metrics']['pattern_accuracy']:.2f}")

if __name__ == "__main__":
    asyncio.run(main())