import React, { useState } from 'react';
import { X, User, Save, Briefcase, Target, Star } from 'lucide-react';

export interface UserProfileData {
  name: string;
  role: string;
  expertise: string[];
  workingStyle: 'detailed' | 'concise' | 'balanced';
  preferences: {
    codeComments: boolean;
    stepByStep: boolean;
    askBeforeAction: boolean;
  };
}

interface UserProfileProps {
  isOpen: boolean;
  onClose: () => void;
  profile: UserProfileData;
  onSave: (profile: UserProfileData) => void;
}

export const UserProfile: React.FC<UserProfileProps> = ({
  isOpen,
  onClose,
  profile,
  onSave
}) => {
  const [formData, setFormData] = useState<UserProfileData>(profile);

  if (!isOpen) return null;

  const handleSave = () => {
    onSave(formData);
    onClose();
  };

  const toggleExpertise = (item: string) => {
    setFormData((prev) => ({
      ...prev,
      expertise: prev.expertise.includes(item)
        ? prev.expertise.filter((e) => e !== item)
        : [...prev.expertise, item]
    }));
  };

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

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
      <div className="bg-atlas-black border-2 border-atlas-green-900 rounded-lg w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-atlas-green-900 sticky top-0 bg-atlas-black">
          <div className="flex items-center gap-3">
            <User size={20} className="text-atlas-yellow-400" />
            <h2 className="text-lg font-bold text-atlas-yellow-400">USER PROFILE</h2>
          </div>
          <button
            onClick={onClose}
            className="text-atlas-green-500 hover:text-atlas-red-400 transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Basic Info */}
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-atlas-cyan-400 mb-3">
              <User size={16} />
              <h3 className="text-sm font-semibold">BASIC INFORMATION</h3>
            </div>

            <div>
              <label className="block text-xs text-atlas-green-700 mb-1">Name</label>
              <input
                type="text"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                className="w-full bg-atlas-green-950/30 border border-atlas-green-900 rounded px-3 py-2 text-sm text-atlas-green-400 outline-none focus:border-atlas-cyan-400"
                placeholder="Your name"
              />
            </div>

            <div>
              <label className="block text-xs text-atlas-green-700 mb-1">Role</label>
              <input
                type="text"
                value={formData.role}
                onChange={(e) => setFormData({ ...formData, role: e.target.value })}
                className="w-full bg-atlas-green-950/30 border border-atlas-green-900 rounded px-3 py-2 text-sm text-atlas-green-400 outline-none focus:border-atlas-cyan-400"
                placeholder="e.g., Senior Developer, Data Scientist"
              />
            </div>
          </div>

          {/* Expertise */}
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-atlas-cyan-400 mb-3">
              <Star size={16} />
              <h3 className="text-sm font-semibold">AREAS OF EXPERTISE</h3>
            </div>

            <div className="grid grid-cols-2 gap-2">
              {expertiseOptions.map((option) => (
                <button
                  key={option}
                  onClick={() => toggleExpertise(option)}
                  className={`px-3 py-2 rounded text-xs text-left transition-all ${
                    formData.expertise.includes(option)
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
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-atlas-cyan-400 mb-3">
              <Briefcase size={16} />
              <h3 className="text-sm font-semibold">WORKING STYLE</h3>
            </div>

            <div className="grid grid-cols-3 gap-2">
              {(['concise', 'balanced', 'detailed'] as const).map((style) => (
                <button
                  key={style}
                  onClick={() => setFormData({ ...formData, workingStyle: style })}
                  className={`px-3 py-2 rounded text-xs capitalize transition-all ${
                    formData.workingStyle === style
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
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-atlas-cyan-400 mb-3">
              <Target size={16} />
              <h3 className="text-sm font-semibold">PREFERENCES</h3>
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
                    checked={formData.preferences[key]}
                    onChange={(e) =>
                      setFormData({
                        ...formData,
                        preferences: { ...formData.preferences, [key]: e.target.checked }
                      })
                    }
                    className="w-4 h-4 accent-atlas-cyan-400"
                  />
                  <span className="text-sm text-atlas-green-500">{label}</span>
                </label>
              ))}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 p-4 border-t border-atlas-green-900 sticky bottom-0 bg-atlas-black">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm text-atlas-green-500 hover:text-atlas-red-400 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="flex items-center gap-2 px-4 py-2 bg-atlas-green-500 text-black rounded font-semibold hover:bg-atlas-cyan-400 transition-colors"
          >
            <Save size={14} />
            Save Profile
          </button>
        </div>
      </div>
    </div>
  );
};
