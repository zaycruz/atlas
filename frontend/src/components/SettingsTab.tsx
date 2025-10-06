import React, { useState } from 'react';
import { Settings, User, Star, Briefcase, Target, Save, Key, Cpu, Eye, EyeOff } from 'lucide-react';
import type { AIModel } from './ModelToggler';
import type { UserProfileData } from './UserProfile';

interface SettingsTabProps {
  profile: UserProfileData;
  onProfileSave: (profile: UserProfileData) => void;
  currentModel: AIModel;
  installedModels: string[];
  availableModels: string[];
  onModelChange: (model: AIModel) => void;
  onAddModel: (model: string) => void;
  modelPullStatus?: {
    model: string | null;
    status: 'idle' | 'started' | 'progress' | 'completed' | 'error';
    message?: string;
  };
  apiKeys: {
    openai: string;
    anthropic: string;
  };
  onApiKeysUpdate: (keys: { openai?: string; anthropic?: string }) => void;
}

const formatModelName = (model: string) =>
  model.replace(/:latest$/i, '').replace(/_/g, ' ').replace(/\b([a-z])/g, (m) => m.toUpperCase());

export const SettingsTab: React.FC<SettingsTabProps> = ({
  profile,
  onProfileSave,
  currentModel,
  installedModels,
  availableModels,
  onModelChange,
  onAddModel,
  modelPullStatus,
  apiKeys,
  onApiKeysUpdate
}) => {
  const [profileData, setProfileData] = useState<UserProfileData>(profile);
  const [localApiKeys, setLocalApiKeys] = useState(apiKeys);
  const [showOpenAIKey, setShowOpenAIKey] = useState(false);
  const [showAnthropicKey, setShowAnthropicKey] = useState(false);
  const [profileChanged, setProfileChanged] = useState(false);
  const [apiKeysChanged, setApiKeysChanged] = useState(false);

  const expertiseOptions = [
    'Frontend Development',
    'Backend Development',
    'DevOps',
    'Data Science',
    'Machine Learning',
    'System Design',
    'Database Design',
    'Security'
  ];

  const toggleExpertise = (item: string) => {
    setProfileData((prev) => ({
      ...prev,
      expertise: prev.expertise.includes(item)
        ? prev.expertise.filter((e) => e !== item)
        : [...prev.expertise, item]
    }));
    setProfileChanged(true);
  };

  const handleProfileSave = () => {
    onProfileSave(profileData);
    setProfileChanged(false);
  };

  const handleApiKeysSave = () => {
    onApiKeysUpdate(localApiKeys);
    setApiKeysChanged(false);
  };

  const addableModels = availableModels.filter((model) => !installedModels.includes(model)).sort();

  return (
    <div className="flex flex-col h-full bg-atlas-black">
      {/* Header */}
      <div className="bg-black px-4 py-3 border-b border-atlas-green-900 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Settings size={16} className="text-atlas-yellow-400" />
          <span className="text-sm font-semibold text-atlas-yellow-400">SETTINGS</span>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-4xl mx-auto space-y-8">
          {/* API Keys Section */}
          <div className="bg-atlas-green-950/20 border border-atlas-green-900 rounded-lg p-6">
            <div className="flex items-center gap-2 text-atlas-cyan-400 mb-4">
              <Key size={18} />
              <h3 className="text-base font-semibold">API KEYS</h3>
            </div>

            <div className="space-y-4">
              {/* OpenAI */}
              <div>
                <label className="block text-xs text-atlas-green-700 mb-2">OpenAI API Key</label>
                <div className="flex gap-2">
                  <div className="relative flex-1">
                    <input
                      type={showOpenAIKey ? 'text' : 'password'}
                      value={localApiKeys.openai}
                      onChange={(e) => {
                        setLocalApiKeys({ ...localApiKeys, openai: e.target.value });
                        setApiKeysChanged(true);
                      }}
                      className="w-full bg-atlas-green-950/30 border border-atlas-green-900 rounded px-3 py-2 text-sm text-atlas-green-400 outline-none focus:border-atlas-cyan-400 pr-10"
                      placeholder="sk-..."
                    />
                    <button
                      onClick={() => setShowOpenAIKey(!showOpenAIKey)}
                      className="absolute right-2 top-1/2 -translate-y-1/2 text-atlas-green-600 hover:text-atlas-green-400"
                    >
                      {showOpenAIKey ? <EyeOff size={16} /> : <Eye size={16} />}
                    </button>
                  </div>
                </div>
                <p className="text-xs text-atlas-green-700 mt-1">
                  Connect your OpenAI account to use GPT models
                </p>
              </div>

              {/* Anthropic */}
              <div>
                <label className="block text-xs text-atlas-green-700 mb-2">Anthropic API Key</label>
                <div className="flex gap-2">
                  <div className="relative flex-1">
                    <input
                      type={showAnthropicKey ? 'text' : 'password'}
                      value={localApiKeys.anthropic}
                      onChange={(e) => {
                        setLocalApiKeys({ ...localApiKeys, anthropic: e.target.value });
                        setApiKeysChanged(true);
                      }}
                      className="w-full bg-atlas-green-950/30 border border-atlas-green-900 rounded px-3 py-2 text-sm text-atlas-green-400 outline-none focus:border-atlas-cyan-400 pr-10"
                      placeholder="sk-ant-..."
                    />
                    <button
                      onClick={() => setShowAnthropicKey(!showAnthropicKey)}
                      className="absolute right-2 top-1/2 -translate-y-1/2 text-atlas-green-600 hover:text-atlas-green-400"
                    >
                      {showAnthropicKey ? <EyeOff size={16} /> : <Eye size={16} />}
                    </button>
                  </div>
                </div>
                <p className="text-xs text-atlas-green-700 mt-1">
                  Connect your Anthropic account to use Claude models
                </p>
              </div>

              {apiKeysChanged && (
                <button
                  onClick={handleApiKeysSave}
                  className="flex items-center gap-2 px-4 py-2 bg-atlas-green-500 text-black rounded font-semibold hover:bg-atlas-cyan-400 transition-colors text-sm"
                >
                  <Save size={14} />
                  Save API Keys
                </button>
              )}
            </div>
          </div>

          {/* Model Selection Section */}
          <div className="bg-atlas-green-950/20 border border-atlas-green-900 rounded-lg p-6">
            <div className="flex items-center gap-2 text-atlas-cyan-400 mb-4">
              <Cpu size={18} />
              <h3 className="text-base font-semibold">MODEL SELECTION</h3>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-xs text-atlas-green-700 mb-2">Current Model</label>
                <div className="text-sm text-atlas-green-400 font-semibold mb-4">
                  {formatModelName(currentModel)}
                </div>
              </div>

              {/* Installed Models */}
              <div>
                <label className="block text-xs text-atlas-green-700 mb-2">Installed Models</label>
                <div className="space-y-2">
                  {installedModels.length === 0 && (
                    <div className="text-xs text-atlas-green-600">No models installed yet.</div>
                  )}
                  {installedModels.map((model) => (
                    <button
                      key={model}
                      onClick={() => onModelChange(model)}
                      className={`w-full text-left px-4 py-2 rounded text-sm transition-colors ${
                        model === currentModel
                          ? 'bg-atlas-green-900/70 text-white border border-atlas-cyan-400'
                          : 'bg-atlas-green-950/30 border border-atlas-green-900 text-atlas-green-400 hover:border-atlas-cyan-400'
                      }`}
                    >
                      {formatModelName(model)}
                    </button>
                  ))}
                </div>
              </div>

              {/* Available Models */}
              {addableModels.length > 0 && (
                <div>
                  <label className="block text-xs text-atlas-green-700 mb-2">Available Models</label>
                  <div className="space-y-2">
                    {addableModels.map((model) => {
                      const isPulling = modelPullStatus?.model === model &&
                        modelPullStatus.status !== 'completed' &&
                        modelPullStatus.status !== 'error';
                      return (
                        <button
                          key={model}
                          onClick={() => onAddModel(model)}
                          disabled={isPulling}
                          className="w-full text-left px-4 py-2 rounded text-sm transition-colors bg-atlas-green-950/30 border border-atlas-green-900 text-atlas-green-400 hover:border-atlas-cyan-400 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          <div className="flex items-center justify-between">
                            <span>{formatModelName(model)}</span>
                            <span className="text-xs text-atlas-cyan-400">
                              {isPulling ? 'Pulling...' : 'Add'}
                            </span>
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}

              {modelPullStatus?.status === 'error' && modelPullStatus.model && (
                <div className="text-xs text-atlas-red-400 mt-2">
                  Failed to pull {formatModelName(modelPullStatus.model)}: {modelPullStatus.message || 'Unknown error'}
                </div>
              )}
            </div>
          </div>

          {/* User Profile Section */}
          <div className="bg-atlas-green-950/20 border border-atlas-green-900 rounded-lg p-6">
            <div className="flex items-center gap-2 text-atlas-cyan-400 mb-4">
              <User size={18} />
              <h3 className="text-base font-semibold">USER PROFILE</h3>
            </div>

            <div className="space-y-6">
              {/* Basic Info */}
              <div className="space-y-4">
                <div>
                  <label className="block text-xs text-atlas-green-700 mb-2">Name</label>
                  <input
                    type="text"
                    value={profileData.name}
                    onChange={(e) => {
                      setProfileData({ ...profileData, name: e.target.value });
                      setProfileChanged(true);
                    }}
                    className="w-full bg-atlas-green-950/30 border border-atlas-green-900 rounded px-3 py-2 text-sm text-atlas-green-400 outline-none focus:border-atlas-cyan-400"
                    placeholder="Your name"
                  />
                </div>

                <div>
                  <label className="block text-xs text-atlas-green-700 mb-2">Role</label>
                  <input
                    type="text"
                    value={profileData.role}
                    onChange={(e) => {
                      setProfileData({ ...profileData, role: e.target.value });
                      setProfileChanged(true);
                    }}
                    className="w-full bg-atlas-green-950/30 border border-atlas-green-900 rounded px-3 py-2 text-sm text-atlas-green-400 outline-none focus:border-atlas-cyan-400"
                    placeholder="e.g., Senior Developer, Data Scientist"
                  />
                </div>
              </div>

              {/* Expertise */}
              <div className="space-y-3">
                <div className="flex items-center gap-2 text-atlas-green-400">
                  <Star size={14} />
                  <label className="text-xs text-atlas-green-700">AREAS OF EXPERTISE</label>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  {expertiseOptions.map((option) => (
                    <button
                      key={option}
                      onClick={() => toggleExpertise(option)}
                      className={`px-3 py-2 rounded text-xs text-left transition-all ${
                        profileData.expertise.includes(option)
                          ? 'bg-atlas-green-500 text-black font-semibold'
                          : 'bg-atlas-green-950/30 border border-atlas-green-900 text-atlas-green-500 hover:border-atlas-cyan-400'
                      }`}
                    >
                      {option}
                    </button>
                  ))}
                </div>
              </div>

              {/* Working Style */}
              <div className="space-y-3">
                <div className="flex items-center gap-2 text-atlas-green-400">
                  <Briefcase size={14} />
                  <label className="text-xs text-atlas-green-700">WORKING STYLE</label>
                </div>
                <div className="grid grid-cols-3 gap-2">
                  {(['concise', 'balanced', 'detailed'] as const).map((style) => (
                    <button
                      key={style}
                      onClick={() => {
                        setProfileData({ ...profileData, workingStyle: style });
                        setProfileChanged(true);
                      }}
                      className={`px-3 py-2 rounded text-xs capitalize transition-all ${
                        profileData.workingStyle === style
                          ? 'bg-atlas-yellow-400 text-black font-semibold'
                          : 'bg-atlas-green-950/30 border border-atlas-green-900 text-atlas-green-500 hover:border-atlas-cyan-400'
                      }`}
                    >
                      {style}
                    </button>
                  ))}
                </div>
              </div>

              {/* Preferences */}
              <div className="space-y-3">
                <div className="flex items-center gap-2 text-atlas-green-400">
                  <Target size={14} />
                  <label className="text-xs text-atlas-green-700">PREFERENCES</label>
                </div>
                <div className="space-y-2">
                  {[
                    { key: 'codeComments' as const, label: 'Include detailed code comments' },
                    { key: 'stepByStep' as const, label: 'Explain step-by-step reasoning' },
                    { key: 'askBeforeAction' as const, label: 'Ask before taking major actions' }
                  ].map(({ key, label }) => (
                    <label key={key} className="flex items-center gap-3 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={profileData.preferences[key]}
                        onChange={(e) => {
                          setProfileData({
                            ...profileData,
                            preferences: { ...profileData.preferences, [key]: e.target.checked }
                          });
                          setProfileChanged(true);
                        }}
                        className="w-4 h-4 accent-atlas-cyan-400"
                      />
                      <span className="text-sm text-atlas-green-500">{label}</span>
                    </label>
                  ))}
                </div>
              </div>

              {profileChanged && (
                <button
                  onClick={handleProfileSave}
                  className="flex items-center gap-2 px-4 py-2 bg-atlas-green-500 text-black rounded font-semibold hover:bg-atlas-cyan-400 transition-colors text-sm"
                >
                  <Save size={14} />
                  Save Profile
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
