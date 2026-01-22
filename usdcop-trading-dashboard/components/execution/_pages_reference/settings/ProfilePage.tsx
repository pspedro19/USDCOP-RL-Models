import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { User, Save } from 'lucide-react'
import { PageHeader } from '@/components/layout/PageHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { useAuthStore } from '@/stores/authStore'
import { authService } from '@/services/authService'
import { toast } from '@/stores/uiStore'
import { RISK_LIMITS } from '@/lib/constants'

export function ProfilePage() {
  const { user, updateUser } = useAuthStore()
  const [riskProfile, setRiskProfile] = useState(user?.risk_profile || 'moderate')

  const updateMutation = useMutation({
    mutationFn: (data: any) => authService.updateProfile(data),
    onSuccess: (data) => {
      updateUser(data)
      toast.success('Profile updated successfully')
    },
    onError: () => {
      toast.error('Failed to update profile')
    },
  })

  const handleSave = () => {
    updateMutation.mutate({ risk_profile: riskProfile })
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Profile"
        breadcrumbs={[
          { label: 'Settings', href: '/settings' },
          { label: 'Profile' },
        ]}
      />

      {/* Profile Info */}
      <Card variant="glass">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <User className="h-5 w-5 text-terminal-accent" />
            Profile Information
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between p-4 rounded-xl bg-slate-800/50">
            <div>
              <p className="text-sm text-text-muted">Email</p>
              <p className="text-text-primary font-medium">{user?.email}</p>
            </div>
            <Badge variant="success">Verified</Badge>
          </div>

          <div className="flex items-center justify-between p-4 rounded-xl bg-slate-800/50">
            <div>
              <p className="text-sm text-text-muted">Subscription</p>
              <p className="text-text-primary font-medium capitalize">
                {user?.subscription_tier} Plan
              </p>
            </div>
            <Button variant="outline" size="sm">Upgrade</Button>
          </div>
        </CardContent>
      </Card>

      {/* Risk Profile */}
      <Card variant="glass">
        <CardHeader>
          <CardTitle>Risk Profile</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {Object.entries(RISK_LIMITS).map(([key, value]) => (
            <label
              key={key}
              className={cn(
                'flex items-center justify-between p-4 rounded-xl border-2 cursor-pointer transition-all',
                riskProfile === key
                  ? 'border-terminal-accent bg-terminal-accent/10'
                  : 'border-slate-700 hover:border-slate-600'
              )}
            >
              <div className="flex items-center gap-3">
                <input
                  type="radio"
                  name="riskProfile"
                  value={key}
                  checked={riskProfile === key}
                  onChange={(e) => setRiskProfile(e.target.value as any)}
                  className="sr-only"
                />
                <div className={cn(
                  'h-4 w-4 rounded-full border-2 flex items-center justify-center',
                  riskProfile === key ? 'border-terminal-accent' : 'border-slate-600'
                )}>
                  {riskProfile === key && (
                    <div className="h-2 w-2 rounded-full bg-terminal-accent" />
                  )}
                </div>
                <div>
                  <p className="font-medium text-text-primary capitalize">{key}</p>
                  <p className="text-sm text-text-muted">{value.description}</p>
                </div>
              </div>
            </label>
          ))}

          <Button
            variant="primary"
            onClick={handleSave}
            isLoading={updateMutation.isPending}
            leftIcon={<Save className="h-4 w-4" />}
            className="w-full mt-4"
          >
            Save Changes
          </Button>
        </CardContent>
      </Card>
    </div>
  )
}

function cn(...classes: (string | boolean | undefined)[]) {
  return classes.filter(Boolean).join(' ')
}
